import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniFMPretrainLoss(nn.Module):
    """Self-supervised loss for Battery-MiniFM pretraining."""

    def __init__(
        self,
        lambda_mask=1.0,
        lambda_future=0.5,
        lambda_freq=0.2,
        lambda_contrast=0.2,
        mask_ratio=0.25,
        eps=1e-6,
        return_dict=True,
    ):
        super().__init__()
        self.lambda_mask = float(lambda_mask)
        self.lambda_future = float(lambda_future)
        self.lambda_freq = float(lambda_freq)
        self.lambda_contrast = float(lambda_contrast)
        self.mask_ratio = float(mask_ratio)
        self.eps = eps
        self.return_dict = bool(return_dict)

    def _masked_reconstruction_loss(self, outputs):
        pred = outputs.get("reconstruct_curve")
        target = outputs.get("target_curve")
        cycle_mask = outputs.get("cycle_mask")
        if pred is None or target is None:
            ref = next(iter(outputs.values()))
            return ref.new_tensor(0.0)

        bsz, seq_len = pred.shape[:2]
        valid = torch.ones(bsz, seq_len, device=pred.device, dtype=pred.dtype)
        if cycle_mask is not None:
            valid = cycle_mask.to(device=pred.device, dtype=pred.dtype)
        token_mask = outputs.get("pretrain_token_mask")
        if token_mask is None:
            token_mask = (torch.rand(bsz, seq_len, device=pred.device) < self.mask_ratio).to(pred.dtype)
        else:
            token_mask = token_mask.to(device=pred.device, dtype=pred.dtype)
        mask = (token_mask * valid).view(bsz, seq_len, 1, 1)
        if mask.sum() < 1:
            mask = valid.view(bsz, seq_len, 1, 1)
        denom = mask.sum().clamp_min(self.eps) * pred.size(-1) * pred.size(-2)
        return (F.smooth_l1_loss(pred, target, reduction="none") * mask).sum() / denom

    def _future_loss(self, outputs):
        pred = outputs.get("future_latent")
        seq = outputs.get("mean_seq")
        if pred is None:
            ref = next(iter(outputs.values()))
            return ref.new_tensor(0.0)
        target_source = outputs.get("target_low_freq")
        if target_source is None:
            target_source = seq
        if target_source is None or target_source.size(1) < 2:
            return pred.new_tensor(0.0)
        steps = min(pred.size(1), target_source.size(1))
        target = target_source[:, -steps:, : pred.size(-1)].detach()
        return F.smooth_l1_loss(pred[:, -steps:, :], target, reduction="mean")

    def _frequency_consistency_loss(self, outputs):
        pred_low = outputs.get("low_freq_latent")
        target_low = outputs.get("target_low_freq")
        if pred_low is None or target_low is None:
            ref = next(iter(outputs.values()))
            return ref.new_tensor(0.0)
        return F.smooth_l1_loss(pred_low, target_low.detach(), reduction="mean")

    def _contrastive_loss(self, outputs):
        z = outputs.get("contrastive_z")
        target_z = outputs.get("target_contrastive_z")
        if z is None or target_z is None:
            ref = next(iter(outputs.values()))
            return ref.new_tensor(0.0)
        return (1.0 - F.cosine_similarity(z, target_z.detach(), dim=-1)).mean()

    def forward(self, outputs, return_dict=None):
        if not isinstance(outputs, dict):
            raise TypeError("MiniFMPretrainLoss expects the BatteryMiniFM output dictionary.")
        mask = self._masked_reconstruction_loss(outputs)
        future = self._future_loss(outputs)
        freq = self._frequency_consistency_loss(outputs)
        contrast = self._contrastive_loss(outputs)
        loss = (
            self.lambda_mask * mask
            + self.lambda_future * future
            + self.lambda_freq * freq
            + self.lambda_contrast * contrast
        )
        result = {
            "loss": loss,
            "mask": mask.detach(),
            "future": future.detach(),
            "freq": freq.detach(),
            "contrast": contrast.detach(),
        }
        should_return_dict = self.return_dict if return_dict is None else bool(return_dict)
        return result if should_return_dict else loss
