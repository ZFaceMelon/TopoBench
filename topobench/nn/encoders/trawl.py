"""Adapter for using existing TopoBench rank-feature encoders before TRAWL."""

from topobench.nn.encoders.base import AbstractFeatureEncoder


class TRAWLFeatureEncoder(AbstractFeatureEncoder):
    """Synchronize a native encoder's x_r outputs with TRAWL state features.

    When using this adapter, set backbone.in_channels to the encoder's output
    widths; input shape inference runs before learned feature encoding.
    """

    def __init__(self, encoder, ranks):
        super().__init__()
        self.encoder, self.ranks = encoder, list(ranks)

    def forward(self, batch):
        batch = self.encoder(batch)
        for rank in self.ranks:
            batch[f"trawl_signal_{rank}"] = batch[f"x_{rank}"]
        return batch
