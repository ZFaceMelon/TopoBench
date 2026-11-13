"""SHA-256 of every random split (train/valid/test indices) under datasets/."""
import glob, hashlib, os, sys
import numpy as np
root = sys.argv[1]
for path in sorted(glob.glob(os.path.join(root, "**/data_splits/**/*.npz"), recursive=True)):
    with np.load(path) as z:
        h = hashlib.sha256()
        for k in ("train", "valid", "test"):
            h.update(k.encode()); h.update(np.asarray(z[k], dtype=np.int64).tobytes())
        sizes = "/".join(str(len(z[k])) for k in ("train", "valid", "test"))
    print(h.hexdigest()[:16], sizes, os.path.relpath(path, root))
