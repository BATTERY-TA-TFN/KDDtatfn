import torch
import torch.nn as nn
import torch.nn.functional as F


class DURALoss(nn.Module):
    """Evidential regression loss for DURA outputs."""

    def __init__(
        self,
        lambda_reg=1e-3,
        lambda_pred=1.0,
        lambda_nll=0.1,
        lambda_mono=0.0,
        lambda_cal=0.0,
        lambda_rel_mono=0.0,
        lambda_hazard_smooth=0.0,
        soh_idx=0,
        target_coverage=0.95,
        temperature=0.01,
        task_weights=None,
        decouple_mean_uncertainty=False,
        return_dict=True,
        eps=1e-6,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_pred = lambda_pred
        self.lambda_nll = lambda_nll
        self.lambda_mono = lambda_mono
        self.lambda_cal = lambda_cal
        self.lambda_rel_mono = lambda_rel_mono
        self.lambda_hazard_smooth = lambda_hazard_smooth
        self.soh_idx = soh_idx
        self.target_coverage = target_coverage
        self.temperature = temperature
        self.task_weights = task_weights
        self.decouple_mean_uncertainty = bool(decouple_mean_uncertainty)
        self.return_dict = return_dict
        self.eps = eps

    def _task_weight_view(self, tensor, task_weights=None):
        weights = self.task_weights if task_weights is None else task_weights
        if weights is None:
            return None
        weights = torch.as_tensor(weights, device=tensor.device, dtype=tensor.dtype).view(1, -1)
        if weights.size(1) != tensor.size(-1):
            if weights.size(1) == 1:
                weights = weights.repeat(1, tensor.size(-1))
            else:
                raise ValueError(f"task_weights length {weights.size(1)} does not match output size {tensor.size(-1)}")
        return weights

    def _weighted_mean(self, values, task_weights=None):
        weights = self._task_weight_view(values, task_weights)
        if weights is None:
            return values.mean()
        while weights.dim() < values.dim():
            weights = weights.unsqueeze(1)
        return (values * weights).sum() / weights.sum().clamp_min(self.eps) / max(1, values.numel() // values.size(-1))

    def nig_nll(self, y, mu, v, alpha, beta, task_weights=None):
        two_blambda = 2.0 * beta * (1.0 + v)
        error_sq = (y - mu) ** 2
        nll = (
            0.5 * torch.log(torch.pi / v.clamp_min(self.eps))
            - alpha * torch.log(two_blambda.clamp_min(self.eps))
            + (alpha + 0.5) * torch.log((v * error_sq + two_blambda).clamp_min(self.eps))
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        return self._weighted_mean(nll, task_weights)

    def evidential_regularizer(self, y, mu, evidence, task_weights=None):
        return self._weighted_mean(torch.abs(y - mu) * evidence, task_weights)

    def prediction_loss(self, y, mu, task_weights=None):
        values = F.smooth_l1_loss(mu, y, reduction="none")
        return self._weighted_mean(values, task_weights)

    def monotonic_loss(self, outputs, y):
        mu_seq = outputs.get("mu_seq") if isinstance(outputs, dict) else None
        if mu_seq is None or mu_seq.dim() != 3 or mu_seq.size(1) < 2:
            return y.new_tensor(0.0)
        soh_idx = min(max(int(self.soh_idx), 0), mu_seq.size(-1) - 1)
        return F.relu(mu_seq[:, 1:, soh_idx] - mu_seq[:, :-1, soh_idx]).mean()

    def calibration_loss(self, y, lower, upper, task_weights=None):
        temp = max(float(self.temperature), self.eps)
        soft_inside_lower = torch.sigmoid((y - lower) / temp)
        soft_inside_upper = torch.sigmoid((upper - y) / temp)
        soft_coverage = self._weighted_mean(soft_inside_lower * soft_inside_upper, task_weights)
        target = y.new_tensor(float(self.target_coverage))
        return F.relu(target - soft_coverage) ** 2

    def reliability_monotonic_loss(self, outputs, y):
        reliability_seq = outputs.get("reliability_seq") if isinstance(outputs, dict) else None
        if reliability_seq is None or reliability_seq.dim() != 3 or reliability_seq.size(1) < 2:
            return y.new_tensor(0.0)
        soh_idx = min(max(int(self.soh_idx), 0), reliability_seq.size(-1) - 1)
        return F.relu(reliability_seq[:, 1:, soh_idx] - reliability_seq[:, :-1, soh_idx]).mean()

    def hazard_smoothness_loss(self, outputs, y):
        hazard_seq = outputs.get("hazard_seq") if isinstance(outputs, dict) else None
        if hazard_seq is None or hazard_seq.dim() != 3 or hazard_seq.size(1) < 3:
            return y.new_tensor(0.0)
        soh_idx = min(max(int(self.soh_idx), 0), hazard_seq.size(-1) - 1)
        hazard = hazard_seq[:, :, soh_idx]
        return torch.abs(hazard[:, 1:] - hazard[:, :-1]).mean()

    def forward(self, outputs, y, weights=None, task_weights=None, return_dict=None):
        if not isinstance(outputs, dict):
            raise TypeError("DURALoss expects the DURA output dictionary.")

        y = y.float()
        mu = outputs["mean"]
        v = outputs["v"]
        alpha = outputs["alpha"]
        beta = outputs["beta"]
        evidence = outputs["evidence"]
        uncertainty_mu = mu.detach() if self.decouple_mean_uncertainty else mu

        pred = self.prediction_loss(y, mu, task_weights)
        nll = self.nig_nll(y, uncertainty_mu, v, alpha, beta, task_weights)
        reg = self.evidential_regularizer(y, uncertainty_mu, evidence, task_weights)
        mono = self.monotonic_loss(outputs, y)
        cal = self.calibration_loss(y, outputs["lower"], outputs["upper"], task_weights)
        rel_mono = self.reliability_monotonic_loss(outputs, y)
        hazard_smooth = self.hazard_smoothness_loss(outputs, y)

        loss = (
            self.lambda_pred * pred
            + self.lambda_nll * nll
            + self.lambda_reg * reg
            + self.lambda_mono * mono
            + self.lambda_cal * cal
            + self.lambda_rel_mono * rel_mono
            + self.lambda_hazard_smooth * hazard_smooth
        )

        if weights is not None:
            # The core NIG loss is already averaged over tasks. This small multiplier
            # preserves the existing sample-weight convention without changing shapes.
            loss = loss * weights.float().mean().clamp_min(self.eps)

        result = {
            "loss": loss,
            "pred": pred.detach(),
            "nll": nll.detach(),
            "reg": reg.detach(),
            "mono": mono.detach(),
            "cal": cal.detach(),
            "rel_mono": rel_mono.detach(),
            "hazard_smooth": hazard_smooth.detach(),
        }
        should_return_dict = self.return_dict if return_dict is None else bool(return_dict)
        return result if should_return_dict else loss
