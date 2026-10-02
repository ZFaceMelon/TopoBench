"""Optional historical reconstruction during supervised fine-tuning."""

from topobench.loss.base import AbstractLoss


class AuxiliaryReconstructionLoss(AbstractLoss):
    """Linearly decay a separately configured reconstruction contribution."""

    def __init__(
        self,
        weight=0.05,
        decay_epochs=20,
        mask_probability=0.0,
        objective="smooth_l1",
    ):
        super().__init__()
        if weight < 0 or decay_epochs < 1 or not 0 <= mask_probability <= 1:
            raise ValueError("Invalid auxiliary reconstruction settings")
        self.weight, self.decay_epochs = weight, decay_epochs
        if objective not in {"mse", "smooth_l1"}:
            raise ValueError("Auxiliary objective must be mse or smooth_l1")

    def forward(self, model_out, batch):
        if "reconstruction_loss" not in model_out:
            return model_out["logits"].sum() * 0
        weight = self.weight * max(
            0, 1 - (model_out["epoch"] - 1) / self.decay_epochs
        )
        return weight * model_out["reconstruction_loss"]
