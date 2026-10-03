"""Categorical cell/constituent encoding and DeepSet walk aggregation."""

import torch
from torch import nn
from torch.nn import functional as F

from topobench.nn.backbones.general.trawl_blocks import SequenceBlock
from topobench.nn.backbones.general.trawl_continuous import (
    ContinuousTRAWL,
)

__all__ = ["CategoricalTRAWL"]


class CategoricalTRAWL(ContinuousTRAWL):
    """Token plus atom/constituent embeddings with normalized PSE fusion.

    Size-two cells sum constituent embeddings; larger cells average them.
    All sequence layers remain configurable. The matching historical-style
    readout uses ``aggregation=deepset`` and temporal mean pooling.
    Regularization follows the historical model: ``layer_dropout`` on layer
    updates plus readout dropout. ``dropout`` only affects the optional
    reconstruction decoder; pooled walk embeddings are not dropped.

    Parameters
    ----------
    vocab_size : int, optional
        Number of cell token types (default: 6).
    num_atom_types : int, optional
        Number of atom/constituent types (default: 64).
    embed_dim : int, optional
        Token and atom embedding width (default: 64).
    pe_dim : int, optional
        Width of the positional/structural encodings (default: 16).
    layer_dropout : float, optional
        Dropout probability on sequence layer updates (default: 0.2).
    **kwargs : dict
        Additional arguments passed to ``ContinuousTRAWL``.
    """

    def __init__(
        self,
        vocab_size=6,
        num_atom_types=64,
        embed_dim=64,
        pe_dim=16,
        layer_dropout=0.2,
        **kwargs,
    ):
        super().__init__(pe_dim=pe_dim, embed_dim=embed_dim, **kwargs)
        del self.continuous_input
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.atom_embed = nn.Embedding(num_atom_types, embed_dim)
        self.pe_norm = nn.LayerNorm(pe_dim)
        self.pe_proj = nn.Linear(pe_dim, pe_dim)
        self.fuse = nn.Linear(embed_dim + pe_dim, self.hidden_dim)
        self.layer_dropout = nn.Dropout(layer_dropout)

    def initialize(self, data_list):
        """Check encoding widths and token vocabulary against the data.

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            Transformed training graphs.
        """
        for data in data_list:
            if data.trawl_pe.shape[1] != self.pe_norm.normalized_shape[0]:
                raise ValueError("Categorical PSE width mismatch")
            if int(data.trawl_tokens.max()) >= self.token_embed.num_embeddings:
                raise ValueError("Token vocabulary is too small")

    def _walk_inputs(self, batch, views=None):
        """Sample walks and gather token, atom and mask features.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch produced by the TRAWL transform.
        views : int, optional
            Number of walk views per graph; see ``ContinuousTRAWL``
            (default: None).

        Returns
        -------
        signals : torch.Tensor
            Concatenated token, atom and atom-mask features along walks.
        pe : torch.Tensor
            Encodings along walks, shape ``[walk, time, pe_dim]``.
        graph_ids : torch.Tensor
            Graph index of each walk.
        paths : torch.Tensor
            Batch-level state indices of shape ``[walk, time]``.
        info : dict
            Host-side walk bookkeeping.
        """
        _, pe, graph_ids, paths, info = super()._walk_inputs(
            batch, views=views
        )
        signals = torch.cat(
            (
                batch.trawl_tokens[paths, None].to(pe),
                batch.trawl_atoms[paths].to(pe),
                batch.trawl_atom_mask[paths],
            ),
            dim=-1,
        )
        return signals, pe, graph_ids, paths, info

    def encode_walks(self, signals, positions, mask_ratio=0.0):
        """Embed categorical walk inputs and encode them.

        Parameters
        ----------
        signals : torch.Tensor
            Token, atom and atom-mask features from ``_walk_inputs``.
        positions : torch.Tensor
            Encodings of shape ``[walk, time, pe_dim]``.
        mask_ratio : float, optional
            Fraction of walk steps masked during training (default: 0.0).

        Returns
        -------
        tokens : torch.Tensor
            Encoded states of shape ``[walk, time, hidden_dim]``.
        pooled : torch.Tensor
            Pooled walk embeddings of shape ``[walk, output_dim]``.
        """
        width = (signals.shape[-1] - 1) // 2
        tokens = signals[..., 0].long()
        atoms = (
            signals[..., 1 : 1 + width]
            .long()
            .clamp(0, self.atom_embed.num_embeddings - 1)
        )
        mask = signals[..., 1 + width :, None]
        summed = (self.atom_embed(atoms) * mask).sum(dim=-2)
        count = mask.sum(dim=-2).clamp_min(1e-6)
        inputs = self.token_embed(tokens) + torch.where(
            count <= 2.5, summed, summed / count
        )
        if mask_ratio and self.training:
            keep = (
                torch.rand(*tokens.shape, 1, device=tokens.device)
                >= mask_ratio
            ).to(inputs)
            inputs, positions = inputs * keep, positions * keep
        x = self.fuse(
            torch.cat((inputs, self.pe_proj(self.pe_norm(positions))), dim=-1)
        )
        for layer in self.encoders[0]:

            def apply_layer(value, layer=layer):
                """Apply one layer with dropout on its residual update.

                Parameters
                ----------
                value : torch.Tensor
                    Walk states of shape ``[walk, time, hidden_dim]``.
                layer : torch.nn.Module
                    Sequence layer to apply.

                Returns
                -------
                torch.Tensor
                    Updated walk states.
                """
                update = (
                    layer(value, residual_only=True)
                    if isinstance(layer, SequenceBlock)
                    else layer(value) - value
                )
                return value + self.layer_dropout(update)

            if self.checkpoint_layers and self.training:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(apply_layer, x, use_reentrant=False)
            else:
                x = apply_layer(x)
        x = self.output_norm(x)
        return x, self._pool(x)

    def reconstruction_loss(self, batch, mask_ratio=0.15):
        """Compute a masked walk-summary reconstruction loss.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch produced by the TRAWL transform.
        mask_ratio : float, optional
            Fraction of walk steps masked before encoding (default: 0.15).

        Returns
        -------
        torch.Tensor
            Mean squared reconstruction error.
        """
        signals, pe, _, paths, _ = self._walk_inputs(batch, views=1)
        _, pooled = self.encode_walks(signals, pe, mask_ratio)
        raw = batch.trawl_continuous[paths]
        delta = (
            (raw[:, 1:] - raw[:, :-1]).abs().mean(1)
            if raw.shape[1] > 1
            else torch.zeros_like(raw[:, 0])
        )
        target = torch.cat((raw.mean(1), pe.mean(1), delta), dim=-1)
        return F.mse_loss(self.reconstruction_decoder(pooled), target)
