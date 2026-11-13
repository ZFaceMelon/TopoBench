"""Re-score a finished TRAWL run's best checkpoint under several walk-view counts.

No retraining. Rebuilds data, splits and model exactly as run.py does from the
run's saved Hydra config, loads the best checkpoint (from trawl_evaluation.json),
and evaluates in eval mode with ``eval_views`` = each requested count on the
validation and test splits, and (optionally) on the training split with the
standard count, so train/val/test are scored under identical settings.

Usage (repo root):
    python scripts/trawl_eval_views.py <run_dir> --views 1,3,8,16 \
        --train-views 3 [--batch-size 64] [--out views.json]
"""

import argparse
import json
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.utils.config_resolvers import register_all_resolvers

register_all_resolvers()


def main():
    """Run the re-scoring and print/save a JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--views", default="1,3,8,16")
    parser.add_argument("--train-views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run = Path(args.run_dir)
    cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
    # The saved config refers to Hydra's runtime; pin those paths explicitly.
    cfg.paths.output_dir = str(run)
    cfg.paths.work_dir = str(Path.cwd())
    report = json.load(open(run / "trawl_evaluation.json"))
    ckpt_path = report["selected_checkpoints"][0]["path"]

    torch.manual_seed(cfg.seed)
    loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = loader.load()
    transforms = (
        hydra.utils.instantiate(cfg.transforms)
        if cfg.get("transforms") is not None
        else None
    )
    pre = PreProcessor(dataset, dataset_dir, transforms)
    train, val, test = pre.load_dataset_splits(cfg.dataset.split_params)
    params = dict(cfg.dataset.get("dataloader_params", {}))
    if args.batch_size:
        params["batch_size"] = args.batch_size
    datamodule = TBDataloader(
        dataset_train=train, dataset_val=val, dataset_test=test, **params
    )
    model = hydra.utils.instantiate(
        cfg.model, evaluator=cfg.evaluator, optimizer=cfg.optimizer, loss=cfg.loss
    )
    model.backbone.initialize(train.data_lst)
    for loss in getattr(getattr(model, "loss", None), "losses", []):
        if getattr(loss, "needs_class_weights", False):
            loss.set_class_weights_from(
                torch.cat([d.y.reshape(-1) for d in train.data_lst]),
                cfg.dataset.parameters.get("num_classes"),
            )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=True)
    trainer = hydra.utils.instantiate(
        cfg.trainer, num_sanity_val_steps=0, logger=False,
        enable_progress_bar=False,
    )

    def score(loader_fn, views):
        for module in model.modules():
            if hasattr(module, "eval_views"):
                module.eval_views = views
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        out = trainer.validate(model=model, dataloaders=loader_fn(), verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.time() - start
        keep = {k.split("/", 1)[1]: float(v) for k, v in out.items()
                if k.split("/", 1)[1] in ("f1", "auroc", "accuracy", "loss")}
        return {**keep, "seconds": round(seconds, 1)}

    results = {"run": str(run), "checkpoint": ckpt_path, "views": {}}
    for views in [int(v) for v in args.views.split(",")]:
        results["views"][views] = {
            "val": score(datamodule.val_dataloader, views),
            "test": score(datamodule.test_dataloader, views),
        }
        print(views, json.dumps(results["views"][views]), flush=True)
    if args.train_views:
        # Same loader construction as val/test, but over the training split
        # (not shuffled, eval mode, no dropout, fixed eval walk seeds).
        train_loader = lambda: TBDataloader(  # noqa: E731
            dataset_train=train, dataset_val=train, dataset_test=test, **params
        ).val_dataloader()
        results["train_eval"] = {
            "views": args.train_views,
            "train": score(train_loader, args.train_views),
        }
        print("train", json.dumps(results["train_eval"]), flush=True)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
