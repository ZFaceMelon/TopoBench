"""Explicit checkpoint averaging and prediction ensembles for recipe studies."""

import copy

import torch
from torch import nn


def average_checkpoints(paths):
    """Average floating parameters/buffers; keep integer buffers from best.

    Paths must be ordered best-first by the validation checkpoint callback.
    Keys, dtypes and shapes must agree. This never averages optimizer states.
    """
    if not paths:
        raise ValueError("No checkpoints available to average")
    result = None
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        if result is None:
            result = {key: value.clone() for key, value in state.items()}
            continue
        if state.keys() != result.keys():
            raise ValueError("Checkpoint keys differ")
        for key, value in state.items():
            if (
                value.shape != result[key].shape
                or value.dtype != result[key].dtype
            ):
                raise ValueError(f"Checkpoint tensor mismatch: {key}")
            if value.is_floating_point() or value.is_complex():
                result[key].add_(value)
    for value in result.values():
        if value.is_floating_point() or value.is_complex():
            value.div_(len(paths))
    return result


class PredictionEnsemble(nn.Module):
    """Run complete model pipelines and average logits before loss/metrics."""

    def __init__(self, model, paths):
        super().__init__()
        members = []
        for path in paths:
            # Copy only neural components, not Lightning trainer/logger state.
            pipeline = nn.ModuleDict(
                {
                    "encoder": copy.deepcopy(model.feature_encoder),
                    "backbone": copy.deepcopy(model.backbone),
                    "readout": copy.deepcopy(model.readout),
                }
            )
            state = torch.load(path, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
            for destination, source in [
                ("encoder", "feature_encoder"),
                ("backbone", "backbone"),
                ("readout", "readout"),
            ]:
                prefix = source + "."
                pipeline[destination].load_state_dict(
                    {
                        key[len(prefix) :]: value
                        for key, value in state.items()
                        if key.startswith(prefix)
                    }
                )
            members.append(pipeline)
        self.members = nn.ModuleList(members)

    def forward(self, batch):
        predictions = []
        for pipeline in self.members:
            member_batch = batch.clone()
            backbone = pipeline["backbone"]
            if hasattr(backbone, "sampling_context"):
                backbone.sampling_context["stage"] = batch.get(
                    "model_state", "Test"
                )
            output = backbone(pipeline["encoder"](member_batch))
            output = pipeline["readout"](output, member_batch)
            predictions.append(output["logits"])
        output["logits"] = torch.stack(predictions).mean(dim=0)
        return output


class PassThroughReadout(nn.Module):
    """Keep the TBModel task interface when logits were produced by an ensemble."""

    def __init__(self, task_level):
        super().__init__()
        self.task_level = task_level

    def forward(self, model_out, batch):
        return model_out
