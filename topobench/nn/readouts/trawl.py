"""Task-independent TRAWL heads compatible with TBModel's loss pipeline."""

import torch
from torch import nn


class TRAWLReadout(nn.Module):
    """Predict from graph embeddings, mean walk logits, or contextual nodes."""

    def __init__(
        self,
        hidden_dim,
        out_channels,
        task_level="graph",
        graph_dim=None,
        head_hidden=None,
        dropout=0.0,
        aggregation="embedding",
        input_dropout=None,
    ):
        super().__init__()
        if task_level not in {"node", "graph"}:
            raise ValueError("TRAWL supports TopoBench's node and graph tasks")
        if aggregation not in {"embedding", "walk_logits", "deepset"}:
            raise ValueError("Unknown readout aggregation")
        self.task_level, self.aggregation = task_level, aggregation
        width = (
            hidden_dim if task_level == "node" else (graph_dim or hidden_dim)
        )
        input_dropout = dropout if input_dropout is None else input_dropout
        if aggregation == "deepset":
            self.phi = nn.Sequential(
                nn.Linear(width, width * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, width),
            )
            self.rho = nn.Sequential(
                nn.Linear(width, width * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, width),
            )
        if head_hidden:
            self.head = nn.Sequential(
                nn.Dropout(input_dropout),
                nn.Linear(width, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_channels),
            )
        else:
            # A linear head has a single dropout, applied to its input.
            self.head = nn.Sequential(
                nn.Dropout(input_dropout), nn.Linear(width, out_channels)
            )

    def forward(self, model_out, batch):
        if self.task_level == "node":
            logits = self.head(model_out["x_0"])
        else:
            if self.aggregation != "embedding" and not model_out.get(
                "walk_readout_compatible", True
            ):
                raise ValueError(
                    "Use embedding aggregation with separate neighborhoods or cell readout to preserve their fusion"
                )
            if self.aggregation == "deepset":
                if "walk_weights" in model_out:
                    raise ValueError(
                        "Branch gating requires embedding or walk_logits aggregation"
                    )
                walks = self.phi(model_out["walk_embedding"])
                views = model_out.get("num_views", 1)
                indices = model_out["walk_batch"] * views + model_out.get(
                    "walk_view", torch.zeros_like(model_out["walk_batch"])
                )
                graph = walks.new_zeros(
                    len(model_out["graph_embedding"]) * views, walks.shape[-1]
                ).index_add(0, indices, walks)
                predictions = self.head(self.rho(graph))
                model_out["logits"] = predictions.reshape(
                    -1, views, predictions.shape[-1]
                ).mean(1)
                return model_out
            if self.aggregation == "walk_logits" and len(
                model_out["walk_batch"]
            ):
                predictions = self.head(model_out["walk_embedding"])
                shape = (
                    len(model_out["graph_embedding"]),
                    predictions.shape[-1],
                )
                if "walk_weights" in model_out:
                    weighted = predictions * model_out["walk_weights"][:, None]
                    model_out["logits"] = weighted.new_zeros(shape).index_add(
                        0,
                        model_out["walk_batch"],
                        weighted,
                    )
                    return model_out
                sums = predictions.new_zeros(shape).index_add(
                    0, model_out["walk_batch"], predictions
                )
                count = predictions.new_zeros(shape[0], 1).index_add(
                    0,
                    model_out["walk_batch"],
                    predictions.new_ones(len(predictions), 1),
                )
                logits = sums / count.clamp_min(1)
                # Host walk counts, when supplied, avoid a device sync.
                walk_counts = model_out.get("walk_counts")
                has_missing = (
                    any(c == 0 for c in walk_counts)
                    if walk_counts is not None
                    else None
                )
                missing = count.squeeze(-1) == 0
                if has_missing or (has_missing is None and missing.any()):
                    logits[missing] = self.head(
                        model_out["graph_embedding"][missing]
                    )
            else:
                logits = self.head(model_out["graph_embedding"])
        model_out["logits"] = logits
        return model_out
