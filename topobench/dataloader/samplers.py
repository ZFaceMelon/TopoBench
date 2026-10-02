"""Optional epoch-indexed sampling for reproducible recipe replay."""

import math

import torch
from torch import distributed
from torch.utils.data import DistributedSampler


def _replicas():
    """Return ``(world_size, rank)`` of an initialized process group."""
    if distributed.is_available() and distributed.is_initialized():
        return distributed.get_world_size(), distributed.get_rank()
    return 1, 0


class EpochRandomSampler(DistributedSampler):
    """Permutation seeded by base seed plus one-based training epoch.

    Subclasses ``DistributedSampler`` so Lightning neither wraps it (the
    wrapper would not forward ``set_epoch``) nor replaces the order. Under
    several processes the shared permutation is padded and sharded by rank.
    """

    def __init__(self, dataset, seed, num_replicas=None, rank=None):
        world, current = _replicas()
        super().__init__(
            dataset,
            num_replicas=world if num_replicas is None else num_replicas,
            rank=current if rank is None else rank,
            shuffle=True,
            seed=int(seed),
            drop_last=False,
        )

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch + 1)
        order = torch.randperm(len(self.dataset), generator=generator).tolist()
        if self.num_replicas == 1:
            return iter(order)
        padding = self.total_size - len(order)
        order += (order * math.ceil(padding / max(len(order), 1)))[:padding]
        return iter(order[self.rank : self.total_size : self.num_replicas])


class UnpaddedDistributedSampler(DistributedSampler):
    """Ordered evaluation shard that visits every sample exactly once.

    ``DistributedSampler`` pads uneven splits with repeated samples, which
    biases synchronized validation and test metrics.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        world, current = _replicas()
        super().__init__(
            dataset,
            num_replicas=world if num_replicas is None else num_replicas,
            rank=current if rank is None else rank,
            shuffle=False,
            drop_last=False,
        )
        self.num_samples = len(
            range(self.rank, len(self.dataset), self.num_replicas)
        )

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return self.num_samples
