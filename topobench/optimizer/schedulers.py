"""Composable schedules for the categorical TRAWL experiments."""

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def warmup_cosine(
    optimizer, epochs, warmup_epochs=10, start_factor=0.01, eta_min=1e-6
):
    """Construct the historical linear-warmup then cosine epoch schedule."""
    if epochs <= warmup_epochs or warmup_epochs < 1:
        raise ValueError("Require 0 < warmup_epochs < epochs")
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=eta_min
            ),
        ],
        milestones=[warmup_epochs],
    )
