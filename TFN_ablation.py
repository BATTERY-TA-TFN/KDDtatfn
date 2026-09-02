"""
TA-TFN 消融实验变体模型
根据论文的三个主要贡献设计消融实验（含“大块头”宏消融）
"""

from torch.nn import Conv1d
import torch.nn.init as init
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from models.TFN import (
    TFconv_STTF, TFN_STTF, CNN, Base_FUNC_CNN,
    StandardTransformerEncoderLayer
)

# ==================== 贡献1消融：任务自适应机制（原小粒度保留以兼容） ====================

class TFconv_STTF_NoTaskAdaptive(TFconv_STTF):
    """
    消融1.1: 移除任务自适应机制
    使用统一的TFconv，不区分SOC和SOH
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, clamp_flag=True, task_type='soc'):
        # 忽略task_type参数，使用统一的TFconv
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, clamp_flag)


class TFconv_STTF_NoTimeBranch(TFconv_STTF):
    """
    消融1.2: 移除时域分支
    只使用频域特征，不进行时频融合
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, clamp_flag=True, task_type='soc'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, clamp_flag)
        self.task_type = task_type

    def forward(self, input):
        # 只返回频域输出，不进行时频融合
        return super().forward(input)


class TFconv_STTF_FixedAlpha(TFconv_STTF):
    """
    消融1.3: 使用固定的alpha值，不进行自适应学习
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, clamp_flag=True, task_type='soc', fixed_alpha=0.5):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, clamp_flag)
        self.task_type = task_type
        self.fixed_alpha = fixed_alpha  # 固定alpha值，不学习
        
        # 时域分支
        self.time_branch = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )
        
        if task_type == 'soh':
            self.soh_lowpass = nn.AvgPool1d(kernel_size=5, stride=1, padding=2)

    def forward(self, input):
        freq_output = super().forward(input)
        time_output = self.time_branch(input)
        
        if self.task_type == 'soh':
            time_output = self.soh_lowpass(time_output)
        
        # 使用固定的alpha值
        output = self.fixed_alpha * freq_output + (1 - self.fixed_alpha) * time_output
        return output


class TFconv_STTF_NoLowpass(TFconv_STTF):
    """
    消融1.4: 移除SOH的低通滤波器
    继承自TFconv_STTF，但添加时域分支和alpha参数，但不添加低通滤波器
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, clamp_flag=True, task_type='soc'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, clamp_flag)
        self.task_type = task_type
        
        # 时域分支
        self.time_branch = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False
        )
        
        if task_type == 'soc':
            self.alpha = nn.Parameter(torch.tensor(0.3))
        else:
            self.alpha = nn.Parameter(torch.tensor(0.9))
        # 不添加低通滤波器（与TFconv_STTF_TaskAdaptive的区别）

    def forward(self, input):
        freq_output = super().forward(input)
        time_output = self.time_branch(input)
        # 不应用低通滤波（与TFconv_STTF_TaskAdaptive的区别）
        
        alpha_clamped = torch.clamp(self.alpha, 0.0, 1.0)
        output = alpha_clamped * freq_output + (1 - alpha_clamped) * time_output
        return output


# ==================== 贡献2消融：参数化时频卷积（原小粒度保留以兼容） ====================

class TFconv_STTF_FixedFreq(TFconv_STTF):
    """
    消融2.1: 使用固定的频率参数，不进行学习
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, clamp_flag=True, fixed_freq=0.1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, clamp_flag)
        self.fixed_freq = fixed_freq
        # 将频率参数设置为固定值，不参与训练
        with torch.no_grad():
            shape = self.superparams.data[:, :, 0].shape
            self.superparams.data[:, :, 0] = torch.ones(shape) * fixed_freq
        # 冻结频率参数
        self.superparams.data[:, :, 0].requires_grad = False

    def _reset_parameters(self):
        # 不重置频率参数，保持固定
        pass


class TFconv_STTF_FixedBandwidth(TFconv_STTF):
    """
    消融2.2: 使用固定的带宽参数，不进行学习
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, clamp_flag=True, fixed_sigma=0.52):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias, clamp_flag)
        self.fixed_sigma = fixed_sigma
        # 将带宽参数设置为固定值
        with torch.no_grad():
            self.superparams.data[:, :, 1] = torch.ones_like(self.superparams.data[:, :, 1]) * fixed_sigma
        # 冻结带宽参数
        self.superparams.data[:, :, 1].requires_grad = False


class TFconv_STTF_NoParameterized(nn.Module):
    """
    消融2.3: 使用标准CNN替代参数化时频卷积
    完全移除参数化机制
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, clamp_flag=True, task_type='soc'):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        # 使用标准初始化
        init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.conv.bias, -bound, bound)

    def forward(self, input):
        return self.conv(input)


# ==================== 宏消融：大块头对照（新增） ====================

class FixedSTFTFrontEnd(nn.Module):
    """固定 STFT 风格前端（不可学习），用余弦窗带通核近似STFT滤波器组。
    输出通道数=out_channels，卷积核大小=kernel_size，padding=kernel_size//2 保持长度。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        # 构造固定滤波器：Hann窗 * cos(2π f_k n)
        n = torch.arange(kernel_size).float()
        window = torch.hann_window(kernel_size)
        # 频率等间距落在(0, 0.5)归一化带宽内
        freqs = torch.linspace(1.0/(2*kernel_size), 0.5 - 1.0/(2*kernel_size), out_channels)
        weight = []
        for f in freqs:
            w = window * torch.cos(2*math.pi*f*n)
            weight.append(w)
        weight = torch.stack(weight, dim=0)  # [out_channels, K]
        weight = weight[:, None, :]  # [out_channels, 1, K]
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        # 冻结参数
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class MacroSharedSinglePath(nn.Module):
    """宏消融A：关闭任务自适应与解耦，单一路共享前端与共享干线。
    使用统一 TFconv_STTF（不区分任务），随后共享 CNN 栈，最终返回同一特征两份。
    """
    def __init__(self, mid_channel: int, seq_len: int):
        super().__init__()
        funckernel_size = mid_channel * 8 - 1
        # 单一路统一 TFconv（不以 task_type区分）
        self.front = TFconv_STTF(in_channels=1, out_channels=mid_channel,
                                 kernel_size=funckernel_size, padding=funckernel_size//2, bias=False)
        # 共享干线（轻量CNN堆叠）
        self.backbone = nn.Sequential(
            nn.Conv1d(mid_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.final_feature_dim = mid_channel

    def forward(self, x):  # x:[N,1,S]
        h = self.front(x)
        h = self.backbone(h).squeeze(-1)  # [N, C]
        return h, h  # 单一路共享 -> 输出两份供 SOC/SOH 头部使用


class MacroFixedSTFTPath(nn.Module):
    """宏消融B1：固定STFT前端 + 共享CNN干线（去除可学习频率/带宽）。"""
    def __init__(self, mid_channel: int, seq_len: int):
        super().__init__()
        funckernel_size = mid_channel * 8 - 1
        self.front = FixedSTFTFrontEnd(in_channels=1, out_channels=mid_channel,
                                       kernel_size=funckernel_size, padding=funckernel_size//2)
        self.backbone = nn.Sequential(
            nn.Conv1d(mid_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel, mid_channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.final_feature_dim = mid_channel

    def forward(self, x):
        h = self.front(x)
        h = self.backbone(h).squeeze(-1)
        return h, h


class MacroCNNOnly(nn.Module):
    """宏消融C2：全CNN（移除自注意力与参数化前端）。"""
    def __init__(self, mid_channel: int, seq_len: int):
        super().__init__()
        self.stem = nn.Conv1d(1, mid_channel, kernel_size=7, padding=3, bias=False)
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel, mid_channel, 5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel, mid_channel, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.final_feature_dim = mid_channel

    def forward(self, x):  # x:[N,1,S]
        h = self.stem(x)
        h = self.body(h).squeeze(-1)
        return h, h


class MacroTransformerOnly(nn.Module):
    """宏消融C1：全Transformer（移除CNN与TF前端）。"""
    def __init__(self, mid_channel: int, seq_len: int, nhead: int = 4, nlayers: int = 2):
        super().__init__()
        d_model = max(32, mid_channel * 2)  # 设定一个稳定的维度
        self.embed = nn.Conv1d(1, d_model, kernel_size=3, padding=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.final_feature_dim = d_model
        self.seq_len = seq_len

    def forward(self, x):  # x:[N,1,S]
        h = self.embed(x)                   # [N, d_model, S]
        h = h.transpose(1, 2)               # [N, S, d_model]
        if h.size(1) != self.pos.size(1):   # 安全检查（可变S时截断/插值）
            if h.size(1) < self.pos.size(1):
                pos = self.pos[:, :h.size(1), :]
            else:
                pos = F.interpolate(self.pos.transpose(1, 2), size=h.size(1), mode='linear', align_corners=False).transpose(1, 2)
        else:
            pos = self.pos
        h = h + pos
        h = self.encoder(h)                 # [N, S, d_model]
        h = h.mean(dim=1)                   # [N, d_model]
        return h, h


# ==================== 完整的消融模型变体（原有） ====================

class TFN_STTF_NoTaskAdaptive(Base_FUNC_CNN):
    """
    消融模型1: 移除任务自适应机制
    使用统一的TFconv，不区分SOC和SOH
    """
    FuncConv1d = TFconv_STTF_NoTaskAdaptive
    
    def __init__(self, mid_channel=16, seq_len=100, **kwargs):
        self.funckernel_size = mid_channel * 8 - 1
        super().__init__(mid_channel=mid_channel, seq_len=seq_len, **kwargs)


class TFN_STTF_NoTimeBranch(Base_FUNC_CNN):
    """
    消融模型2: 移除时域分支
    只使用频域特征
    """
    FuncConv1d = TFconv_STTF_NoTimeBranch
    
    def __init__(self, mid_channel=16, seq_len=100, **kwargs):
        self.funckernel_size = mid_channel * 8 - 1
        super().__init__(mid_channel=mid_channel, seq_len=seq_len, **kwargs)


class TFN_STTF_FixedAlpha(Base_FUNC_CNN):
    """
    消融模型3: 使用固定的alpha值
    """
    FuncConv1d = None  # 需要特殊处理
    
    def __init__(self, mid_channel=16, seq_len=100, fixed_alpha=0.5, **kwargs):
        self.funckernel_size = mid_channel * 8 - 1
        self.cnn_backbone = CNN(mid_channel=mid_channel, seq_len=seq_len)
        self.final_feature_dim = self.cnn_backbone.output_dim
        
        funckernel_size = self.funckernel_size
        clamp_flag = kwargs.get('clamp_flag', True)
        
        # 为SOC和SOH分别创建固定alpha的funconv
        self.funconv_soc = TFconv_STTF_FixedAlpha(
            in_channels=1, out_channels=mid_channel,
            kernel_size=funckernel_size, padding=funckernel_size // 2,
            clamp_flag=clamp_flag, task_type='soc', fixed_alpha=fixed_alpha
        )
        self.funconv_soh = TFconv_STTF_FixedAlpha(
            in_channels=1, out_channels=mid_channel,
            kernel_size=funckernel_size, padding=funckernel_size // 2,
            clamp_flag=clamp_flag, task_type='soh', fixed_alpha=fixed_alpha
        )
        self.superparams = None

    def forward(self, x):
        x_soc = self.funconv_soc(x)
        soc_features, _ = self.cnn_backbone(x_soc)
        x_soh = self.funconv_soh(x)
        _, soh_features = self.cnn_backbone(x_soh)
        return soc_features, soh_features

    def getweight(self):
        weight_soc = self.funconv_soc.weight.cpu().detach().numpy()
        superparams_soc = self.funconv_soc.superparams.cpu().detach().numpy()
        return weight_soc, superparams_soc


class TFN_STTF_NoLowpass(Base_FUNC_CNN):
    """
    消融模型4: 移除SOH的低通滤波器
    """
    FuncConv1d = None
    
    def __init__(self, mid_channel=16, seq_len=100, **kwargs):
        self.funckernel_size = mid_channel * 8 - 1
        self.cnn_backbone = CNN(mid_channel=mid_channel, seq_len=seq_len)
        self.final_feature_dim = self.cnn_backbone.output_dim
        
        funckernel_size = self.funckernel_size
        clamp_flag = kwargs.get('clamp_flag', True)
        
        self.funconv_soc = TFconv_STTF_NoLowpass(
            in_channels=1, out_channels=mid_channel,
            kernel_size=funckernel_size, padding=funckernel_size // 2,
            clamp_flag=clamp_flag, task_type='soc'
        )
        self.funconv_soh = TFconv_STTF_NoLowpass(
            in_channels=1, out_channels=mid_channel,
            kernel_size=funckernel_size, padding=funckernel_size // 2,
            clamp_flag=clamp_flag, task_type='soh'
        )
        self.superparams = None

    def forward(self, x):
        x_soc = self.funconv_soc(x)
        soc_features, _ = self.cnn_backbone(x_soc)
        x_soh = self.funconv_soh(x)
        _, soh_features = self.cnn_backbone(x_soh)
        return soc_features, soh_features

    def getweight(self):
        weight_soc = self.funconv_soc.weight.cpu().detach().numpy()
        superparams_soc = self.funconv_soc.superparams.cpu().detach().numpy()
        return weight_soc, superparams_soc


class TFN_STTF_FixedFreq(Base_FUNC_CNN):
    """
    消融模型5: 使用固定的频率参数
    """
    FuncConv1d = TFconv_STTF_FixedFreq
    
    def __init__(self, mid_channel=16, seq_len=100, fixed_freq=0.1, **kwargs):
        self.funckernel_size = mid_channel * 8 - 1
        self.cnn_backbone = CNN(mid_channel=mid_channel, seq_len=seq_len)
        self.final_feature_dim = self.cnn_backbone.output_dim
        
        funckernel_size = self.funckernel_size
        clamp_flag = kwargs.get('clamp_flag', True)
        
        self.funconv = TFconv_STTF_FixedFreq(
            in_channels=1, out_channels=mid_channel,
            kernel_size=funckernel_size, padding=funckernel_size // 2,
            clamp_flag=clamp_flag, fixed_freq=fixed_freq
        )
        self.superparams = self.funconv.superparams

    def forward(self, x):
        x = self.funconv(x)
        soc_features, soh_features = self.cnn_backbone(x)
        return soc_features, soh_features

    def getweight(self):
        weight = self.funconv.weight.cpu().detach().numpy()
        superparams = self.funconv.superparams.cpu().detach().numpy()
        return weight, superparams


class TFN_STTF_StandardCNN(Base_FUNC_CNN):
    """
    消融模型6: 使用标准CNN替代参数化时频卷积
    """
    FuncConv1d = TFconv_STTF_NoParameterized
    
    def __init__(self, mid_channel=16, seq_len=100, **kwargs):
        self.funckernel_size = mid_channel * 8 - 1
        self.cnn_backbone = CNN(mid_channel=mid_channel, seq_len=seq_len)
        self.final_feature_dim = self.cnn_backbone.output_dim
        
        funckernel_size = self.funckernel_size
        clamp_flag = kwargs.get('clamp_flag', True)
        
        self.funconv = TFconv_STTF_NoParameterized(
            in_channels=1, out_channels=mid_channel,
            kernel_size=funckernel_size, padding=funckernel_size // 2,
            bias=False, clamp_flag=clamp_flag, task_type='soc'
        )
        self.superparams = None

    def forward(self, x):
        x = self.funconv(x)
        soc_features, soh_features = self.cnn_backbone(x)
        return soc_features, soh_features

    def getweight(self):
        weight = self.funconv.conv.weight.cpu().detach().numpy()
        return weight, None


# ==================== 主模型类（用于消融实验） ====================

class Model(nn.Module):
    """
    消融实验的主模型类
    支持不同的TFN变体（含宏消融：macro_*）
    """
    def __init__(self, configs, ablation_type='full'):
        """
        Args:
            configs: 配置对象
            ablation_type: 消融类型（放大到架构/参数级别）
                - 'full': 完整TA-TFN (baseline)
                - 宏消融（强对照）：
                    * 'macro_M1_off_single_path'：关闭任务自适应与解耦，单一路共享
                    * 'macro_M2_off_fixed_stft'：固定STFT前端，移除可学习频率/带宽
                    * 'macro_M3_transformer_only'：全Transformer
                    * 'macro_M3_cnn_only'：全CNN
                - 兼容保留：
                    * 'shared_tfconv' / 'shared_alpha' / 'fixed_alpha' / 'swap_alpha'
                    * 'freq_only' / 'time_only'
                    * 'freeze_f' / 'freeze_sigma' / 'tie_f' / 'tie_sigma'
                    * 'no_lowpass' / 'fixed_freq' / 'standard_cnn'
                    * 'no_task_adaptive' / 'no_time_branch'
        """
        super(Model, self).__init__()
        self.configs = configs
        self.ablation_type = ablation_type
        
        if configs.c_out == 2:
            self.c_out_soc = 1
            self.c_out_soh = 1
        elif configs.c_out == 1:
            raise ValueError("configs.c_out must represent total output dimension for SOC and SOH.")
        else:
            self.c_out_soc = configs.c_out // 2
            self.c_out_soh = configs.c_out - self.c_out_soc

        # 先构建 full 版本（便于统一在其上做“共享/冻结/替换”）
        from models.TFN import TFN_STTF_TaskAdaptive
        self.intra_TFN = TFN_STTF_TaskAdaptive(mid_channel=configs.mid_channel)

        # === 宏消融（大块头）优先分支 ===
        if ablation_type == 'macro_M1_off_single_path':
            self.intra_TFN = MacroSharedSinglePath(mid_channel=configs.mid_channel, seq_len=getattr(configs, 'seq_len', 100))
        elif ablation_type == 'macro_M2_off_fixed_stft':
            self.intra_TFN = MacroFixedSTFTPath(mid_channel=configs.mid_channel, seq_len=getattr(configs, 'seq_len', 100))
        elif ablation_type == 'macro_M3_transformer_only':
            self.intra_TFN = MacroTransformerOnly(mid_channel=configs.mid_channel, seq_len=getattr(configs, 'seq_len', 100))
        elif ablation_type == 'macro_M3_cnn_only':
            self.intra_TFN = MacroCNNOnly(mid_channel=configs.mid_channel, seq_len=getattr(configs, 'seq_len', 100))

        # === 其余类型：在 full 基础上修改或替换 ===
        elif ablation_type == 'shared_tfconv':
            # 两路共用同一个 funconv（含 superparams/alpha）
            self.intra_TFN.funconv_soh = self.intra_TFN.funconv_soc
            if hasattr(self.intra_TFN.funconv_soc, 'alpha'):
                self.intra_TFN.funconv_soh.alpha = self.intra_TFN.funconv_soc.alpha

        elif ablation_type == 'shared_alpha':
            if hasattr(self.intra_TFN.funconv_soc, 'alpha') and hasattr(self.intra_TFN.funconv_soh, 'alpha'):
                self.intra_TFN.funconv_soh.alpha = self.intra_TFN.funconv_soc.alpha

        elif ablation_type == 'fixed_alpha':
            fa_soc = getattr(configs, 'fixed_alpha_soc', 0.3)
            fa_soh = getattr(configs, 'fixed_alpha_soh', 0.9)
            if hasattr(self.intra_TFN.funconv_soc, 'alpha'):
                with torch.no_grad():
                    self.intra_TFN.funconv_soc.alpha.copy_(torch.tensor(float(fa_soc)))
                self.intra_TFN.funconv_soc.alpha.requires_grad = False
            if hasattr(self.intra_TFN.funconv_soh, 'alpha'):
                with torch.no_grad():
                    self.intra_TFN.funconv_soh.alpha.copy_(torch.tensor(float(fa_soh)))
                self.intra_TFN.funconv_soh.alpha.requires_grad = False

        elif ablation_type == 'swap_alpha':
            if hasattr(self.intra_TFN.funconv_soc, 'alpha') and hasattr(self.intra_TFN.funconv_soh, 'alpha'):
                with torch.no_grad():
                    a_soc = self.intra_TFN.funconv_soc.alpha.clone()
                    a_soh = self.intra_TFN.funconv_soh.alpha.clone()
                    self.intra_TFN.funconv_soc.alpha.copy_(a_soh)
                    self.intra_TFN.funconv_soh.alpha.copy_(a_soc)
                self.intra_TFN.funconv_soc.alpha.requires_grad = False
                self.intra_TFN.funconv_soh.alpha.requires_grad = False

        elif ablation_type == 'freq_only':
            for fun in [self.intra_TFN.funconv_soc, self.intra_TFN.funconv_soh]:
                if hasattr(fun, 'alpha'):
                    with torch.no_grad():
                        fun.alpha.copy_(torch.tensor(1.0))
                    fun.alpha.requires_grad = False

        elif ablation_type == 'time_only':
            for fun in [self.intra_TFN.funconv_soc, self.intra_TFN.funconv_soh]:
                if hasattr(fun, 'alpha'):
                    with torch.no_grad():
                        fun.alpha.copy_(torch.tensor(0.0))
                    fun.alpha.requires_grad = False

        elif ablation_type == 'freeze_f':
            for fun in [self.intra_TFN.funconv_soc, self.intra_TFN.funconv_soh]:
                if hasattr(fun, 'superparams'):
                    fun.superparams.requires_grad = True
                    fun.superparams.register_hook(lambda g: torch.cat((torch.zeros_like(g[:, :, :1]), g[:, :, 1:]), dim=-1))

        elif ablation_type == 'freeze_sigma':
            for fun in [self.intra_TFN.funconv_soc, self.intra_TFN.funconv_soh]:
                if hasattr(fun, 'superparams'):
                    fun.superparams.requires_grad = True
                    fun.superparams[:, :, 1].requires_grad = False
                    fun.superparams.register_hook(lambda g: torch.cat((g[:, :, :1], torch.zeros_like(g[:, :, 1:])), dim=-1))

        elif ablation_type == 'tie_f':
            soc, soh = self.intra_TFN.funconv_soc, self.intra_TFN.funconv_soh
            if hasattr(soc, 'superparams') and hasattr(soh, 'superparams'):
                soh.superparams[:, :, 0].requires_grad = False
                def _align_f(m, inp):
                    with torch.no_grad():
                        soh.superparams[:, :, 0].copy_(soc.superparams[:, :, 0])
                self._pre_hook = soc.register_forward_pre_hook(_align_f)

        elif ablation_type == 'tie_sigma':
            soc, soh = self.intra_TFN.funconv_soc, self.intra_TFN.funconv_soh
            if hasattr(soc, 'superparams') and hasattr(soh, 'superparams'):
                soh.superparams[:, :, 1].requires_grad = False
                def _align_sigma(m, inp):
                    with torch.no_grad():
                        soh.superparams[:, :, 1].copy_(soc.superparams[:, :, 1])
                self._pre_hook = soc.register_forward_pre_hook(_align_sigma)

        elif ablation_type == 'no_lowpass':
            # 移除 SOH 低通
            if hasattr(self.intra_TFN.funconv_soh, 'soh_lowpass'):
                self.intra_TFN.funconv_soh.soh_lowpass = nn.Identity()

        elif ablation_type == 'fixed_freq':
            self.intra_TFN = TFN_STTF_FixedFreq(mid_channel=configs.mid_channel,
                                               fixed_freq=getattr(configs, 'fixed_freq', 0.1))
        elif ablation_type == 'standard_cnn':
            self.intra_TFN = TFN_STTF_StandardCNN(mid_channel=configs.mid_channel)
        elif ablation_type == 'no_task_adaptive':
            self.intra_TFN = TFN_STTF_NoTaskAdaptive(mid_channel=configs.mid_channel)
        elif ablation_type == 'no_time_branch':
            self.intra_TFN = TFN_STTF_NoTimeBranch(mid_channel=configs.mid_channel)
        # 其余类型已在上方基于 full 做修改

        if hasattr(self.intra_TFN, 'superparams'):
            self.superparams = self.intra_TFN.superparams
        else:
            self.superparams = None

        self.d_model = self.intra_TFN.final_feature_dim

        self.projection_soc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model // 2, self.c_out_soc)
        )
        self.projection_soh = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model // 2, self.c_out_soh)
        )

        self.apply(self._init_weights)

        # 打印可训练参数，便于核对消融是否真正生效
        trainable = [n for n,p in self.named_parameters() if p.requires_grad]
        print(f'[Ablation={ablation_type}] Trainable params: {len(trainable)}')
        for n in trainable[:80]:
            print('  ', n)
        if len(trainable) > 80:
            print('  ...')

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x_enc, *args, **kwargs):
        B, C, S = x_enc.shape
        intra_input = x_enc.reshape(-1, 1, S)  # [B*C, 1, S]

        def safe_check_tensor(name, x):
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"⚠️ NaN/Inf detected after {name}")
                x = torch.nan_to_num(x, nan=1e-6, posinf=1e-6, neginf=-1e-6)
            return x

        def safe_forward(name, layer, x):
            try:
                x = layer(x)
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
                raise
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"⚠️ NaN/Inf detected after {name}")
                print(f"   stats -> mean={x.mean().item():.6f}, "
                      f"std={x.std().item():.6f}, "
                      f"min={x.min().item():.6f}, "
                      f"max={x.max().item():.6f}")
                x = torch.nan_to_num(x, nan=1e-6, posinf=1e-6, neginf=-1e-6)
            return x

        try:
            soc_output, soh_output = self.intra_TFN(intra_input)
        except Exception as e:
            print(f"❌ Error in intra_TFN: {e}")
            raise

        soc_output = safe_check_tensor("intra_TFN_SOC", soc_output)
        soh_output = safe_check_tensor("intra_TFN_SOH", soh_output)

        soc_cycle_features = soc_output.reshape(B, C, self.d_model)  # [B, C, d]
        soc_pooled_output = soc_cycle_features.permute(0, 2, 1)  # [B, d, C]
        soc_pooled_output = F.adaptive_avg_pool1d(soc_pooled_output, 1).squeeze(-1)

        soh_cycle_features = soh_output.reshape(B, C, self.d_model)  # [B, C, d]
        soh_pooled_output = soh_cycle_features.permute(0, 2, 1)  # [B, d, C]
        soh_pooled_output = F.adaptive_avg_pool1d(soh_pooled_output, 1).squeeze(-1)

        preds_soc = safe_forward("projection_soc", self.projection_soc, soc_pooled_output)
        preds_soh = safe_forward("projection_soh", self.projection_soh, soh_pooled_output)

        preds = torch.cat([preds_soc, preds_soh], dim=-1)

        return preds
