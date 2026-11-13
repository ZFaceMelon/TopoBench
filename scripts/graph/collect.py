"""Collect val/test accuracy, epochs and timings from run dirs."""
import csv, glob, json, os, re, sys
rows = []
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*"))):
    if not os.path.isfile(os.path.join(d, "EXIT")):
        continue
    name = os.path.basename(d)
    code = open(os.path.join(d, "EXIT")).read().strip()
    log = open(os.path.join(d, "run.log"), errors="ignore").read()
    m = re.search(r"seconds=(\d+)", log)
    secs = int(m.group(1)) if m else None
    val = test = None
    ev = os.path.join(d, "trawl_evaluation.json")
    metrics = glob.glob(os.path.join(d, "csv/*/metrics.csv"))
    last = {}
    epochs = 0
    train_times = []
    if metrics:
        for r in csv.DictReader(open(metrics[0])):
            for k, v in r.items():
                if v != "":
                    last[k] = v
            if r.get("epoch"):
                epochs = max(epochs, int(float(r["epoch"])) + 1)
    if os.path.isfile(ev):
        mt = json.load(open(ev))["metrics"]
        val, test = mt.get("val_best_rerun/accuracy"), mt.get("test_best_rerun/accuracy")
    else:
        val = last.get("val_best_rerun/accuracy") or last.get("best_epoch/val/accuracy")
        test = last.get("test_best_rerun/accuracy") or last.get("test/accuracy")
    # per-epoch train time from progress bars "Epoch k: 100%|...| n/n [mm:ss<"
    for mm, ss in re.findall(r"Epoch \d+: 100%\|[^|]*\| \d+/\d+ \[(\d+):(\d+)<", log):
        train_times.append(int(mm) * 60 + int(ss))
    tpe = sum(train_times[-20:]) / max(1, len(train_times[-20:])) if train_times else None
    rows.append((name, code, val, test, epochs, secs, tpe))
for r in rows:
    print("\t".join("" if x is None else (("%.4f" % float(x)) if isinstance(x, (float, str)) and re.fullmatch(r"[0-9.]+", str(x)) and "." in str(x) else str(x)) for x in r))
