# Preview dataset files under DATA_ROOT by printing a pandas-style head
# for many common formats (CSV/TSV, Parquet/Feather, Pickle, NumPy, MAT, PT).
# Usage:
#   python scripts/preview_dataframes.py --list
#   python scripts/preview_dataframes.py --index 3
#   python scripts/preview_dataframes.py --path C:\path\to\file.csv --head 10
# Optional: use --key to preview specific keys in NPZ/MAT/PT files.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# -----------------------------
# Edit these defaults as needed.
# -----------------------------
DATA_ROOT = Path(r"C:\Users\yourpath\TopoBench\data")
# If you prefer a default file, set this to a relative path under DATA_ROOT.
DEFAULT_RELATIVE_PATH: Optional[str] = None
# Or set a default index from the file list (use --list to see indices).
DEFAULT_INDEX: Optional[int] = None

DATA_FILE_EXTS = {
    ".csv",
    ".tsv",
    ".txt",
    ".parquet",
    ".feather",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".pt",
    ".mat",
}


def iter_data_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in DATA_FILE_EXTS:
                yield Path(dirpath) / filename


def list_data_files(root: Path) -> list[Path]:
    return sorted(iter_data_files(root))


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def truncate_repr(value: object, max_len: int = 200) -> str:
    text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def coerce_array(value: object) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return None


def preview_dataframe(df: pd.DataFrame, head: int, max_cols: int) -> None:
    print(f"DataFrame shape: {df.shape}")
    with pd.option_context("display.max_columns", max_cols, "display.width", 160):
        print(df.head(head))


def describe_object(value: object) -> str:
    length = None
    try:
        length = len(value)  # type: ignore[arg-type]
    except Exception:
        length = None
    if length is not None:
        return f"type={type(value)} len={length}"
    return f"type={type(value)}"


def to_preview_dataframe(value: object) -> Optional[pd.DataFrame]:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, dict):
        rows = []
        for k, v in value.items():
            arr = coerce_array(v)
            if arr is not None:
                rows.append(
                    {
                        "key": k,
                        "type": type(v).__name__,
                        "shape": getattr(arr, "shape", None),
                        "dtype": getattr(arr, "dtype", None),
                    }
                )
            else:
                rows.append(
                    {
                        "key": k,
                        "type": type(v).__name__,
                        "shape": None,
                        "dtype": None,
                        "value": truncate_repr(v),
                    }
                )
        return pd.DataFrame(rows)
    if isinstance(value, (list, tuple)):
        # If it's a list of lists (or tuples) with uniform length, treat as table.
        if value and all(isinstance(row, (list, tuple)) for row in value):
            try:
                return pd.DataFrame(value)
            except Exception:
                pass
        rows = [(i, truncate_repr(v)) for i, v in enumerate(value)]
        return pd.DataFrame(rows, columns=["index", "value"])
    return None


def preview_array(name: str, array: np.ndarray, head: int, max_cols: int) -> None:
    print_header(f"{name} | shape={array.shape} dtype={array.dtype}")

    if array.ndim == 0:
        print(array.item())
        return

    if array.ndim == 1:
        df = pd.DataFrame({"value": array})
        preview_dataframe(df, head, max_cols)
        return

    if array.ndim == 2:
        cols = [f"col_{i}" for i in range(array.shape[1])]
        df = pd.DataFrame(array, columns=cols)
        preview_dataframe(df, head, max_cols)
        return

    # For higher dimensions, flatten all but the first axis for preview.
    reshaped = array.reshape(array.shape[0], -1)
    cols = [f"col_{i}" for i in range(reshaped.shape[1])]
    df = pd.DataFrame(reshaped, columns=cols)
    preview_dataframe(df, head, max_cols)


def preview_object(name: str, value: object, head: int, max_cols: int) -> None:
    df = to_preview_dataframe(value)
    if df is not None:
        print_header(f"{name} | {describe_object(value)}")
        preview_dataframe(df, head, max_cols)
        # For dicts with array-like values, also show a head for each tensor.
        if isinstance(value, dict):
            for k, v in value.items():
                arr = coerce_array(v)
                if arr is not None:
                    preview_array(str(k), np.asarray(arr), head, max_cols)
        return
    print_header(f"{name} | {describe_object(value)}")
    print(truncate_repr(value))


def preview_npz(path: Path, head: int, max_cols: int, keys: Optional[list[str]]) -> None:
    data = np.load(path, allow_pickle=True)
    all_keys = list(data.keys())
    print_header(f"NPZ keys ({len(all_keys)}): {all_keys}")

    for key in all_keys:
        if keys and key not in keys:
            continue
        array = data[key]
        preview_array(key, np.asarray(array), head, max_cols)


def preview_npy(path: Path, head: int, max_cols: int) -> None:
    array = np.load(path, allow_pickle=True)
    preview_array(path.name, np.asarray(array), head, max_cols)


def preview_csv_like(path: Path, head: int, max_cols: int, sep: Optional[str]) -> None:
    df = pd.read_csv(path, sep=sep)
    preview_dataframe(df, head, max_cols)


def preview_parquet(path: Path, head: int, max_cols: int) -> None:
    df = pd.read_parquet(path)
    preview_dataframe(df, head, max_cols)


def preview_pickle(path: Path, head: int, max_cols: int) -> None:
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        preview_dataframe(obj, head, max_cols)
        return
    if isinstance(obj, pd.Series):
        preview_dataframe(obj.to_frame(), head, max_cols)
        return
    print_header(f"Pickle object {describe_object(obj)}")
    arr = coerce_array(obj)
    if arr is not None:
        preview_array(path.name, arr, head, max_cols)
        return
    preview_object(path.name, obj, head, max_cols)


def preview_mat(path: Path, head: int, max_cols: int, keys: Optional[list[str]]) -> None:
    try:
        import h5py

        is_h5 = h5py.is_hdf5(path)
    except Exception:
        is_h5 = False

    if is_h5:
        import h5py

        with h5py.File(path, "r") as f:
            all_keys = list(f.keys())
            print_header(f"MAT v7.3 (HDF5) keys ({len(all_keys)}): {all_keys}")
            for key in all_keys:
                if keys and key not in keys:
                    continue
                obj = f[key]
                if not isinstance(obj, h5py.Dataset):
                    continue
                array = np.asarray(obj)
                preview_array(key, array, head, max_cols)
        return

    from scipy.io import loadmat

    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    all_keys = [k for k in data.keys() if not k.startswith("__")]
    print_header(f"MAT keys ({len(all_keys)}): {all_keys}")

    for key in all_keys:
        if keys and key not in keys:
            continue
        value = data[key]
        arr = coerce_array(value)
        if arr is not None:
            preview_array(key, np.asarray(arr), head, max_cols)
            continue
        preview_object(key, value, head, max_cols)


def preview_pt(path: Path, head: int, max_cols: int, keys: Optional[list[str]]) -> None:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required to preview .pt files") from exc

    obj = torch.load(path, map_location="cpu")
    print_header(f"Loaded .pt object type: {type(obj)}")

    # Torch Geometric Data has .keys() and attribute access for tensors.
    if hasattr(obj, "keys") and callable(obj.keys):
        all_keys = list(obj.keys)
        print_header(f"PT keys ({len(all_keys)}): {all_keys}")
        for key in all_keys:
            if keys and key not in keys:
                continue
            value = getattr(obj, key)
            arr = coerce_array(value)
            if arr is not None:
                preview_array(key, arr, head, max_cols)
            else:
                preview_object(key, value, head, max_cols)
        return

    if isinstance(obj, dict):
        all_keys = list(obj.keys())
        print_header(f"PT dict keys ({len(all_keys)}): {all_keys}")
        for key in all_keys:
            if keys and key not in keys:
                continue
            value = obj[key]
            arr = coerce_array(value)
            if arr is not None:
                preview_array(str(key), arr, head, max_cols)
            else:
                preview_object(str(key), value, head, max_cols)
        return

    if isinstance(obj, (list, tuple)):
        print_header(f"PT sequence length: {len(obj)}")
        for idx, value in enumerate(obj):
            key = f"item_{idx}"
            if keys and key not in keys:
                continue
            arr = coerce_array(value)
            if arr is not None:
                preview_array(key, arr, head, max_cols)
            else:
                preview_object(key, value, head, max_cols)
        return

    arr = coerce_array(obj)
    if arr is not None:
        preview_array(path.name, arr, head, max_cols)
        return

    preview_object(path.name, obj, head, max_cols)


def resolve_target(args: argparse.Namespace) -> Path:
    if args.path:
        return Path(args.path)

    data_files = list_data_files(DATA_ROOT)
    if not data_files:
        raise RuntimeError(f"No data files found under {DATA_ROOT}")

    if args.index is not None:
        if args.index < 0 or args.index >= len(data_files):
            raise IndexError(f"Index {args.index} out of range (0..{len(data_files) - 1})")
        return data_files[args.index]

    if DEFAULT_RELATIVE_PATH:
        return DATA_ROOT / DEFAULT_RELATIVE_PATH

    if DEFAULT_INDEX is not None:
        if DEFAULT_INDEX < 0 or DEFAULT_INDEX >= len(data_files):
            raise IndexError(
                f"DEFAULT_INDEX {DEFAULT_INDEX} out of range (0..{len(data_files) - 1})"
            )
        return data_files[DEFAULT_INDEX]

    # Fallback to the first file to keep it runnable without edits.
    return data_files[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preview dataset files as pandas-style heads."
    )
    parser.add_argument("--path", help="Absolute path to a data file.")
    parser.add_argument(
        "--index", type=int, help="Index from --list output (0-based)."
    )
    parser.add_argument(
        "--list", action="store_true", help="List data files under DATA_ROOT."
    )
    parser.add_argument(
        "--head", type=int, default=5, help="Number of rows to show."
    )
    parser.add_argument(
        "--max-cols", type=int, default=20, help="Max columns to display."
    )
    parser.add_argument(
        "--sep", default=None, help="CSV separator (defaults to auto)."
    )
    parser.add_argument(
        "--key",
        action="append",
        dest="keys",
        help="Limit preview to specific keys (repeatable).",
    )

    args = parser.parse_args()

    if args.list:
        data_files = list_data_files(DATA_ROOT)
        print_header(f"Data files under {DATA_ROOT} ({len(data_files)})")
        for i, path in enumerate(data_files):
            print(f"[{i:03d}] {path}")
        return 0

    target = resolve_target(args)
    if not target.exists():
        raise FileNotFoundError(target)

    print_header(f"Previewing: {target}")

    ext = target.suffix.lower()
    if ext in {".csv"}:
        preview_csv_like(target, args.head, args.max_cols, args.sep)
        return 0
    if ext in {".tsv", ".txt"}:
        sep = args.sep if args.sep is not None else "\t"
        preview_csv_like(target, args.head, args.max_cols, sep)
        return 0
    if ext in {".parquet", ".feather"}:
        preview_parquet(target, args.head, args.max_cols)
        return 0
    if ext in {".pkl", ".pickle"}:
        preview_pickle(target, args.head, args.max_cols)
        return 0
    if ext == ".npz":
        preview_npz(target, args.head, args.max_cols, args.keys)
        return 0
    if ext == ".npy":
        preview_npy(target, args.head, args.max_cols)
        return 0
    if ext == ".mat":
        preview_mat(target, args.head, args.max_cols, args.keys)
        return 0
    if ext == ".pt":
        preview_pt(target, args.head, args.max_cols, args.keys)
        return 0

    raise ValueError(f"Unsupported file extension: {ext}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}")
        raise