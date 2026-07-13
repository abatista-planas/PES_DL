"""Run the MIT-BIH anomaly-detection benchmark and print a comparison table.

Tracks:
  1. supervised discriminator      -- trained on real labeled beats (upper bound)
  2. train-on-normal discriminator -- trained on normal + synthetic anomalies (BeatGAN-comparable)
  3. PyOD baselines                -- fit on normal beats only
  4. BeatGAN (cited)               -- AUROC 0.945 / F1 0.816 (IJCAI 2019), not re-run

Usage:
  python -m pes_1D.anomaly.benchmark --data-dir ./data/mitdb --epochs 20
  python -m pes_1D.anomaly.benchmark --max-records 6 --epochs 5   # quick subset
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from pes_1D.anomaly import baselines, detector, mitbih
from pes_1D.anomaly.ecg_anomalies import corrupt_batch

# BeatGAN reported on MIT-BIH (Zhou et al., IJCAI 2019). Cited, not re-run.
BEATGAN_CITED = {"AUROC": 0.945, "AUPRC": None, "F1": 0.816}


def _sanitize(scores: np.ndarray) -> np.ndarray:
    """Replace non-finite scores while preserving the ranking of finite ones."""
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    hi = finite.max() if finite.size else 1.0
    lo = finite.min() if finite.size else 0.0
    med = float(np.median(finite)) if finite.size else 0.0
    return np.nan_to_num(scores, nan=med, posinf=hi, neginf=lo)


def compute_metrics(scores: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """AUROC, AUPRC (average precision), and best-threshold F1 (higher score = anomaly)."""
    y = np.asarray(y).astype(int)
    scores = _sanitize(scores)
    auroc = float(roc_auc_score(y, scores))
    auprc = float(average_precision_score(y, scores))
    prec, rec, _ = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return {"AUROC": auroc, "AUPRC": auprc, "F1": float(np.max(f1))}


def run(
    data_dir: str,
    epochs: int = 50,
    channels: int = 3,
    split: str = "beat",
    hidden=None,
    max_records: int | None = None,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    np.random.seed(seed)
    mitbih.download(data_dir)
    if split == "patient":
        X_train, y_train, X_test, y_test = mitbih.load_ds1_ds2(
            data_dir, max_records=max_records
        )
    else:
        X_train, y_train, X_test, y_test = mitbih.load_beat_level(
            data_dir, seed=seed, max_records=max_records
        )
    print(
        f"split={split} channels={channels} epochs={epochs} | "
        f"train beats: {len(X_train)} ({int((y_train == 0).sum())} normal / "
        f"{int((y_train == 1).sum())} abnormal) | "
        f"test beats: {len(X_test)} ({int((y_test == 0).sum())} normal / "
        f"{int((y_test == 1).sum())} abnormal)"
    )
    grid = X_train.shape[1]
    X_train_normal = X_train[y_train == 0]

    results: Dict[str, Dict[str, float]] = {}

    # Track 1 -- supervised discriminator (real labels)
    m = detector.train_discriminator(
        X_train, y_train, grid_size=grid, channels=channels,
        hidden_channels=hidden, epochs=epochs, seed=seed,
    )
    results["discriminator (supervised)"] = compute_metrics(
        detector.evaluate_scores(m, X_test, channels=channels), y_test
    )

    # Track 2 -- train-on-normal discriminator (synthetic anomalies)
    X_anom = corrupt_batch(X_train_normal, seed=seed)
    Xd = np.concatenate([X_train_normal, X_anom])
    yd = np.concatenate([np.zeros(len(X_train_normal)), np.ones(len(X_anom))])
    m2 = detector.train_discriminator(
        Xd, yd, grid_size=grid, channels=channels,
        hidden_channels=hidden, epochs=epochs, seed=seed,
    )
    results["discriminator (train-on-normal)"] = compute_metrics(
        detector.evaluate_scores(m2, X_test, channels=channels), y_test
    )

    # Track 3 -- PyOD baselines (fit on normal beats only)
    for name, scores in baselines.run_pyod_baselines(X_train_normal, X_test, seed).items():
        try:
            results[name] = compute_metrics(scores, y_test)
        except Exception as exc:  # a broken baseline shouldn't kill the run
            print(f"[benchmark] metrics for {name} failed: {type(exc).__name__}: {exc}")

    # Track 4 -- cited reference
    results["BeatGAN (cited)"] = dict(BEATGAN_CITED)
    return results


def format_table(results: Dict[str, Dict[str, float]]) -> str:
    def cell(v):
        return "   -  " if v is None else f"{v:.4f}"

    rows = ["method                            AUROC    AUPRC    F1",
            "-" * 60]
    for name, m in results.items():
        rows.append(
            f"{name:<32} {cell(m.get('AUROC'))}  {cell(m.get('AUPRC'))}  {cell(m.get('F1'))}"
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="./data/mitdb")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--channels", type=int, default=3, choices=[1, 3])
    ap.add_argument("--split", default="beat", choices=["beat", "patient"])
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./results/mitbih_benchmark.csv")
    args = ap.parse_args()

    results = run(
        args.data_dir,
        epochs=args.epochs,
        channels=args.channels,
        split=args.split,
        max_records=args.max_records,
        seed=args.seed,
    )
    table = format_table(results)
    print("\n" + table)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("method,AUROC,AUPRC,F1\n")
        for name, m in results.items():
            f.write(
                f"{name},{m.get('AUROC')},{m.get('AUPRC')},{m.get('F1')}\n"
            )
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
