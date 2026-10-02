"""Regenerate small numerical fixtures from a trusted local TRAWL checkout.

Usage: python scripts/trawl/build_reference_fixture.py --source-root ../
Only selected mathematical classes/functions are executed; original training
scripts are never imported. The generated fixture has no pickle objects.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from torch.nn import functional as F


def definitions(path, names):
    source = path.read_text()
    tree = ast.parse(source)
    return "\n\n".join(
        ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in names
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test/nn/trawl/fixtures/reference.npz"),
    )
    args = parser.parse_args()
    snapshot = (
        args.source_root / "snapshots/proteins_peak224_l5_mean_7470_5seed"
    )
    pure = snapshot / "mutag_edvw_ccmamba.py"
    model_source = snapshot / "trawl_proteins.py"
    sisa_source = (
        args.source_root
        / "snapshots/proteins_peak224_richfeat_hybrid_learnable_gate/trawl_proteins.py"
    )
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
        "os": os,
        "math": math,
        "checkpoint": checkpoint,
        "D_STATE": 4,
        "D_CONV": 4,
        "EXPAND": 2,
        "PRETRAIN_MASK_RATIO": 0.15,
    }
    exec(definitions(pure, {"PureTorchMambaBlock"}), namespace)
    namespace["build_mamba_block"] = lambda **kwargs: namespace[
        "PureTorchMambaBlock"
    ](**kwargs)
    exec(
        definitions(model_source, {"MambaBlock", "BiophysicalTRAWLMamba"}),
        namespace,
    )
    torch.manual_seed(781)
    reference = namespace["BiophysicalTRAWLMamba"](
        4, 6, 1, 4, 16, 4, 2, dropout=0.0
    ).eval()
    signals, pe = torch.randn(2, 3, 7, 4), torch.randn(2, 3, 7, 6)
    values = {"signals": signals.numpy(), "pe": pe.numpy()}
    with torch.no_grad():
        values["pooled"] = reference.encode_pooled(signals, pe).numpy()
        values["targets"] = reference.reconstruction_targets(
            signals, pe
        ).numpy()
    for key, tensor in reference.state_dict().items():
        values["continuous::" + key] = tensor.numpy().copy()
    reference.encode_pooled(signals, pe).square().mean().backward()
    for key, parameter in reference.named_parameters():
        if parameter.grad is not None:
            values["continuous_grad::" + key] = parameter.grad.numpy().copy()
    torch.optim.Adam(reference.parameters(), lr=1e-4, weight_decay=1e-3).step()
    for key, parameter in reference.named_parameters():
        if parameter.grad is not None:
            values["continuous_step::" + key] = (
                parameter.detach().numpy().copy()
            )
    exec(
        definitions(
            sisa_source,
            {"build_rope_cache", "apply_rope", "SISALayer", "SISABlock"},
        ),
        namespace,
    )
    torch.manual_seed(782)
    sisa = namespace["SISABlock"](16, n_heads=2, d_ssm=4).eval()
    tokens = torch.randn(2, 7, 16)
    values["sisa_input"] = tokens.numpy()
    with torch.no_grad():
        values["sisa_output"] = sisa(tokens).numpy()
    for key, tensor in sisa.state_dict().items():
        values["sisa::" + key] = tensor.numpy().copy()
    sisa(tokens).square().mean().backward()
    for key, parameter in sisa.named_parameters():
        if parameter.grad is not None:
            values["sisa_grad::" + key] = parameter.grad.numpy().copy()
    torch.optim.Adam(sisa.parameters(), lr=1e-4, weight_decay=1e-3).step()
    for key, parameter in sisa.named_parameters():
        if parameter.grad is not None:
            values["sisa_step::" + key] = parameter.detach().numpy().copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **values)
    metadata = {
        "sources": {
            str(path.relative_to(args.source_root)).replace(
                "\\", "/"
            ): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [pure, model_source, sisa_source]
        },
        "torch": torch.__version__,
        "seed": [781, 782],
        "scope": "Layer/continuous-encoder forward, gradient and Adam-step parity; not end-to-end run reproduction.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
