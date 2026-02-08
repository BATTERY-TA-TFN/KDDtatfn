import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import evaluate
import torch.nn.functional as F
from scipy.signal.windows import gaussian
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import time
from torch import nn
import wandb

plt.switch_backend('agg')


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, 'Percent': trainable_num / total_num}


class Augment_time_series_family(object):
    '''
    This is a set of augmentation for methods for time series
    '''
    def __init__(self, n_holes, mean=0, std=0.02):
        pass


class Downsample_Expand_aug(object):
    '''
    '''
    def __init__(self, rate=0.1):
        pass


class Masking_aug(object):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate

    def __call__(self, seq):
        '''
        Params:
            seq: Tensor sequence of size (B, num_var, L)
        '''
        seq = F.dropout(seq, self.drop_rate)
        return seq


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    lr_adjust = {}  # <--- 在函数开头添加此行
    if args.lradj == 'type1':
        if epoch > args.least_epochs:
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - args.least_epochs) // 1))}
        else:
            lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print(f'{args.lradj}| Updating learning rate to {lr}')
            else:
                print(f'{args.lradj}| Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True, least_epochs=5):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode
        self.least_epochs = least_epochs

    def __call__(self, epoch, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            if epoch > self.least_epochs:
                # the early stopping won't count before some epochs are trained
                self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            self.accelerator.save_model(model, path)
            self.accelerator.print(f'The checkpoint is saved in {path}!')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)


# 计算均方根误差 (RMSE)
def root_mean_squared_error(y_true, y_pred):
    """计算RMSE"""
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def evaluate_battery_predictions(args, accelerator, model, data_loader, data_set, criterion, task_type='multi'):
    """
    评估电池SOC、SOH预测性能

    Args:
        args: 配置参数
        accelerator: accelerator对象
        model: 预测模型
        data_loader: 数据加载器
        data_set: 数据集对象
        criterion: 损失函数
        task_type: 预测任务类型 ('SOC', 'SOH', 'multi')

    Returns:
        dict: 包含各种评估指标的字典
    """
    model.eval()
    total_loss = 0.0

    # 存储预测和真实值
    if task_type == 'multi':
        soc_preds, soc_targets = [], []
        soh_preds, soh_targets = [], []
    else:
        preds, targets = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            cycle_curve_data, curve_attn_mask, labels, weights, battery_type_id = batch

            # 前向传播
            outputs = model(cycle_curve_data, battery_type_id, curve_attn_mask)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # 计算损失
            if hasattr(criterion, '__call__'):
                loss = criterion(outputs, labels)
            else:
                loss = F.mse_loss(outputs, labels)
            total_loss += loss.item()

            # 反标准化预测和标签
            if task_type == 'multi':
                # 多任务预测
                outputs_np = outputs.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()

                # 假设outputs和labels的shape为[batch_size, 3]，分别对应SOC、SOH
                scalers = data_set.return_label_scalers()

                # SOC
                soc_pred_denorm = scalers['soc'].inverse_transform(outputs_np[:, 0].reshape(-1, 1)).flatten()
                soc_target_denorm = scalers['soc'].inverse_transform(labels_np[:, 0].reshape(-1, 1)).flatten()
                soc_preds.extend(soc_pred_denorm)
                soc_targets.extend(soc_target_denorm)

                # SOH
                soh_pred_denorm = scalers['soh'].inverse_transform(outputs_np[:, 1].reshape(-1, 1)).flatten()
                soh_target_denorm = scalers['soh'].inverse_transform(labels_np[:, 1].reshape(-1, 1)).flatten()
                soh_preds.extend(soh_pred_denorm)
                soh_targets.extend(soh_target_denorm)


            else:
                # 单任务预测
                scaler = data_set.return_label_scalers()[task_type.lower()]
                pred_denorm = scaler.inverse_transform(outputs.detach().cpu().numpy().reshape(-1, 1)).flatten()
                target_denorm = scaler.inverse_transform(labels.detach().cpu().numpy().reshape(-1, 1)).flatten()
                preds.extend(pred_denorm)
                targets.extend(target_denorm)

    avg_loss = total_loss / len(data_loader)

    # 计算评估指标
    results = {'loss': avg_loss}

    if task_type == 'multi':
        # 多任务评估
        tasks = ['SOC', 'SOH']
        all_preds = [soc_preds, soh_preds]
        all_targets = [soc_targets, soh_targets]

        for task, pred_list, target_list in zip(tasks, all_preds, all_targets):
            if len(pred_list) > 0:
                mae = mean_absolute_error(target_list, pred_list)
                rmse = root_mean_squared_error(target_list, pred_list)
                mape = mean_absolute_percentage_error(target_list, pred_list)

                results[f'{task}_MAE'] = mae
                results[f'{task}_RMSE'] = rmse
                results[f'{task}_MAPE'] = mape

                if accelerator:
                    accelerator.print(f"{task} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

    else:
        # 单任务评估
        if len(preds) > 0:
            mae = mean_absolute_error(targets, preds)
            rmse = root_mean_squared_error(targets, preds)
            mape = mean_absolute_percentage_error(targets, preds)

            results.update({
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,

            })

            if accelerator:
                accelerator.print(f"{task_type} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

    model.train()
    return results


def vali_battery_prediction(args, accelerator, model, vali_data, vali_loader, criterion, task_type='multi'):
    """
    验证电池预测模型性能的简化版本

    Args:
        args: 配置参数
        accelerator: accelerator对象
        model: 预测模型
        vali_data: 验证数据集
        vali_loader: 验证数据加载器
        criterion: 损失函数
        task_type: 预测任务类型

    Returns:
        tuple: (rmse, mae, mape) 或 多任务的话返回字典
    """
    results = evaluate_battery_predictions(args, accelerator, model, vali_loader, vali_data, criterion, task_type)

    if task_type == 'multi':
        # 多任务情况下，返回平均指标
        avg_mae = np.mean([results.get(f'{task}_MAE', 0) for task in ['SOC', 'SOH'] if f'{task}_MAE' in results])
        avg_rmse = np.mean(
            [results.get(f'{task}_RMSE', 0) for task in ['SOC', 'SOH'] if f'{task}_RMSE' in results])
        avg_mape = np.mean(
            [results.get(f'{task}_MAPE', 0) for task in ['SOC', 'SOH'] if f'{task}_MAPE' in results])
        return avg_rmse, avg_mae, avg_mape
    else:
        return results['RMSE'], results['MAE'], results['MAPE']


def test_battery_prediction(args, accelerator, model, test_data, test_loader, criterion, task_type='multi'):
    """
    测试电池预测模型性能

    Args:
        args: 配置参数
        accelerator: accelerator对象
        model: 预测模型
        test_data: 测试数据集
        test_loader: 测试数据加载器
        criterion: 损失函数
        task_type: 预测任务类型

    Returns:
        dict: 详细的评估结果
    """
    if accelerator:
        accelerator.print("\n" + "=" * 50)
        accelerator.print("🔧 开始电池预测模型测试...")
        accelerator.print("=" * 50)

    results = evaluate_battery_predictions(args, accelerator, model, test_loader, test_data, criterion, task_type)

    if accelerator:
        accelerator.print("\n" + "=" * 50)
        accelerator.print("📊 电池预测测试结果:")
        accelerator.print("=" * 50)

        if task_type == 'multi':
            for task in ['SOC', 'SOH']:
                if f'{task}_MAE' in results:
                    accelerator.print(f"\n{task} 预测结果:")
                    accelerator.print(f"  MAE:  {results[f'{task}_MAE']:.4f}")
                    accelerator.print(f"  RMSE: {results[f'{task}_RMSE']:.4f}")
                    accelerator.print(f"  MAPE: {results[f'{task}_MAPE']:.2f}%")
        else:
            accelerator.print(f"\n{task_type} 预测结果:")
            accelerator.print(f"  MAE:  {results['MAE']:.4f}")
            accelerator.print(f"  RMSE: {results['RMSE']:.4f}")
            accelerator.print(f"  MAPE: {results['MAPE']:.2f}%")

        accelerator.print("=" * 50)

    return results


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('../dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content


# utils/tools.py 文件末尾添加内容
# --------------------------------------------------------------------------------------------------

def set_seed(seed):
    """设置所有可能的随机种子以确保实验可复现性。"""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import random
        random.seed(seed)


def data_provider_baseline(args, flag, label_scalers=None):
    """
    数据加载器的占位符函数，用于模拟主训练脚本所需的 Data_loader 导入。

    注意：在实际训练脚本中，你需要导入并使用你定义的 DataLoader 类（如 Dataset_Custom）。
    """
    # 这是一个占位符。在实际代码中，你需要根据 args.dataset 导入和实例化你的数据集和加载器。
    print(f"Warning: Using placeholder data_provider_baseline for flag='{flag}'.")

    # 假设你有一个 BatteryDataset 类和 torch.utils.data.DataLoader
    # from data_provider.data_factory import data_provider # 实际项目中应该导入这个

    # 返回示例结构，以避免主脚本崩溃：
    class DummyDataset:
        def __init__(self, scalers=None):
            self.label_scalers = scalers

        def return_label_scalers(self):
            return self.label_scalers

    dummy_data = DummyDataset(scalers=label_scalers)
    # 返回一个空的 DataLoader (会使训练循环立即结束)
    dummy_loader = torch.utils.data.DataLoader(dummy_data, batch_size=args.batch_size)

    return dummy_data, dummy_loader

# --------------------------------------------------------------------------------------------------