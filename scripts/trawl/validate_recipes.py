"""Validate full-size recipe updates on real data, without claiming score parity.

Run from the repository root with ``python -m scripts.trawl.validate_recipes``.
Results include the resolved configuration and distinguish a small data sample
from a full benchmark run. Dataset loaders use their ordinary local caches.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from topobench.dataloader import DataloadDataset
from topobench.dataloader.utils import collate_fn
from topobench.transforms.data_manipulations.trawl_historical import (
    HistoricalCellTransform,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--graphs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recipes",
        nargs="+",
        default=[
            "proteins_mamba",
            "proteins_hybrid",
            "nci1_hybrid",
            "nci1_sisa",
            "proteins_gated",
            "zinc_categorical",
        ],
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    results = {
        "torch": torch.__version__,
        "device": args.device,
        "scope": "Real-data forward/backward and optimizer update; no score reproduction claim",
        "results": [],
    }
    for recipe in args.recipes:
        start = time.perf_counter()
        with initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parents[2] / "configs"),
            version_base="1.3",
        ):
            cfg = compose(
                config_name="run",
                overrides=[
                    f"experiment=trawl/{recipe}",
                    "logger=[]",
                    f"paths.output_dir={args.output.resolve().parent.as_posix()}",
                    f"paths.log_dir={args.output.resolve().parent.as_posix()}",
                    f"paths.work_dir={Path.cwd().as_posix()}",
                ],
            )
            torch.manual_seed(cfg.seed)
            dataset, _ = instantiate(cfg.dataset.loader).load()
            transform_options = OmegaConf.to_container(
                cfg.transforms.historical, resolve=True
            )
            transform = HistoricalCellTransform(**transform_options)
            graphs = [
                transform.build(dataset[i], int(cfg.seed) + i)
                for i in range(args.graphs)
            ]
            model = instantiate(
                cfg.model,
                evaluator=cfg.evaluator,
                optimizer=cfg.optimizer,
                loss=cfg.loss,
            )
            model.backbone.initialize(graphs)
            model.to(args.device).train()
            wrapped = DataloadDataset(graphs)
            batch = collate_fn([wrapped[i] for i in range(len(wrapped))]).to(
                args.device
            )
            optimizer = model.optimizer.configure_optimizer(
                model.parameters()
            )["optimizer"]
            before = (
                next(model.backbone.encoders.parameters()).detach().clone()
            )
            output = model(batch)
            loss_output = model.loss(model_out=output, batch=batch)
            loss = loss_output["loss"]
            loss.backward()
            gradients = [
                p.grad for p in model.parameters() if p.grad is not None
            ]
            assert torch.isfinite(loss) and gradients
            assert all(torch.isfinite(g).all() for g in gradients)
            optimizer.step()
            assert not torch.equal(
                before, next(model.backbone.encoders.parameters())
            )
            row = {
                "recipe": recipe,
                "graphs": args.graphs,
                "loss": float(loss.detach()),
                "parameters": sum(p.numel() for p in model.parameters()),
                "seconds": time.perf_counter() - start,
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            results["results"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
            print(
                f"PASS {recipe}: {row['parameters']} parameters, {row['seconds']:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
