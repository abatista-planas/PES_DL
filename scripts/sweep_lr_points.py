"""Sweep the number of observed low-res points (4..16) and compare models.

For each N in {4, 6, 8, 10, 12, 14, 16} every method reconstructs the
high-resolution PES (128 points) from the same N observed points, and we
report the RMSE on the full high-res grid (energies normalized to [-1, 1]).

Methods: linear interpolation, cubic spline, GP (RBF), GP (Matern 5/2),
supervised CNN (MSE only), and the fixed conditional GAN (CNN + adversarial
fine-tuning).  An out-of-family test on the Reudenberg O2 curve is included
to check that the learned prior generalizes beyond LJ/Morse.

Usage:  python scripts/sweep_lr_points.py [outdir] [placement]

placement: "uniform" (default) — evenly spaced observed points;
"irregular" — per-curve random placements (endpoints kept, interior
points drawn without replacement, so clusters and gaps occur). In
irregular mode the networks train on per-curve random placements and
the O2 curve is evaluated over 32 random placements.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pes_1D.superres import (
    ConditionalDiscriminator,
    RefineGenerator,
    build_conditioning,
    lr_indices,
    predict_cubic_spline,
    predict_gp,
    predict_linear,
    predict_nn,
    random_lr_indices,
    rmse_per_curve,
    sample_dataset,
    sample_pes_curve,
    train_gan,
    train_supervised,
)

HR_SIZE = 128
FAMILIES = ("lennard_jones", "morse", "buckingham_exp6", "extended_rydberg")
N_TRAIN_PER_FAMILY = 500
N_TEST_PER_FAMILY = 100
N_SWEEP = [4, 6, 8, 10, 12, 14, 16]
EPOCHS_SUPERVISED = 40
EPOCHS_GAN = 25
SEED = 7

# Optional GPU-scale multiplier: set env PES_SCALE=N to grow the training set
# and epochs N-fold (default 1 keeps parity with the committed CPU results).
SCALE = int(os.environ.get("PES_SCALE", "1"))


def resolve_device() -> str:
    forced = os.environ.get("PES_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(outdir: Path, placement: str = "uniform") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    n_train_per_family = N_TRAIN_PER_FAMILY * SCALE
    epochs_sup = EPOCHS_SUPERVISED * SCALE
    epochs_gan = EPOCHS_GAN * SCALE
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    print(
        f"Device: {device} | scale x{SCALE} "
        f"(train {len(FAMILIES) * n_train_per_family}/run, "
        f"epochs sup={epochs_sup} gan={epochs_gan})"
    )
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    idx_rng = np.random.default_rng(SEED + 10)

    print(f"Generating data for families: {FAMILIES} ...")
    train_hr, _ = sample_dataset(rng, n_train_per_family, HR_SIZE, FAMILIES)
    test_hr, test_fam = sample_dataset(
        np.random.default_rng(SEED + 1), N_TEST_PER_FAMILY, HR_SIZE, FAMILIES
    )
    o2_hr = sample_pes_curve(
        np.random.default_rng(SEED + 2), "reudenberg_o2", HR_SIZE,
        wall_factor=2.0, tail_factor=0.05,
    )[None, :]
    target_train = torch.from_numpy(train_hr).unsqueeze(1)

    rows = []
    gan_health = {}
    for n_lr in N_SWEEP:
        t0 = time.time()
        if placement == "uniform":
            idx_train = idx_test = lr_indices(HR_SIZE, n_lr)
            o2_eval = o2_hr
            idx_o2 = idx_train
        else:  # per-curve irregular placements; O2 over 32 placements
            idx_train = random_lr_indices(idx_rng, HR_SIZE, n_lr, len(train_hr))
            idx_test = random_lr_indices(idx_rng, HR_SIZE, n_lr, len(test_hr))
            o2_eval = np.repeat(o2_hr, 32, axis=0)
            idx_o2 = random_lr_indices(idx_rng, HR_SIZE, n_lr, len(o2_eval))
        print(f"\n===== N = {n_lr} observed points ({placement}) =====")

        # --- classical baselines (per test curve) ---
        preds = {
            "linear": predict_linear(test_hr, idx_test),
            "cubic_spline": predict_cubic_spline(test_hr, idx_test),
            "gp_rbf": predict_gp(test_hr, idx_test, "rbf"),
            "gp_matern": predict_gp(test_hr, idx_test, "matern"),
        }
        o2_preds = {
            "linear": predict_linear(o2_eval, idx_o2),
            "cubic_spline": predict_cubic_spline(o2_eval, idx_o2),
            "gp_rbf": predict_gp(o2_eval, idx_o2, "rbf"),
            "gp_matern": predict_gp(o2_eval, idx_o2, "matern"),
        }

        # --- neural models ---
        cond_train = build_conditioning(train_hr, idx_train)
        cond_test = build_conditioning(test_hr, idx_test)
        cond_o2 = build_conditioning(o2_eval, idx_o2)

        gen_sup = RefineGenerator()
        train_supervised(
            gen_sup, cond_train, target_train,
            epochs=epochs_sup, seed=SEED, verbose=True, device=device,
        )
        preds["cnn_supervised"] = predict_nn(gen_sup, cond_test)
        o2_preds["cnn_supervised"] = predict_nn(gen_sup, cond_o2)

        gen_gan = copy.deepcopy(gen_sup)  # SRGAN recipe: adversarial fine-tune
        disc = ConditionalDiscriminator()
        log = train_gan(
            gen_gan, disc, cond_train, target_train,
            epochs=epochs_gan, seed=SEED, verbose=True, device=device,
        )
        preds["gan"] = predict_nn(gen_gan, cond_test)
        o2_preds["gan"] = predict_nn(gen_gan, cond_o2)

        k = max(len(log.d_real_acc) - 100, 0)
        gan_health[n_lr] = {
            "d_real_acc_tail": float(np.mean(log.d_real_acc[k:])),
            "d_fake_acc_tail": float(np.mean(log.d_fake_acc[k:])),
            "g_adv_tail": float(np.mean(log.g_adv[k:])),
            "g_recon_tail": float(np.mean(log.g_recon[k:])),
        }

        for name, p in preds.items():
            e = rmse_per_curve(p, test_hr)
            row = dict(
                n_lr=n_lr, model=name,
                rmse_mean=float(e.mean()), rmse_median=float(np.median(e)),
                rmse_p90=float(np.percentile(e, 90)),
                rmse_o2=float(rmse_per_curve(o2_preds[name], o2_eval).mean()),
            )
            for fam in FAMILIES:
                row[f"rmse_{fam}"] = float(e[test_fam == fam].mean())
            rows.append(row)
        print(f"  done in {time.time() - t0:.0f}s")
        for r in rows[-len(preds):]:
            print(
                f"    {r['model']:<15} rmse {r['rmse_mean']:.3e} "
                f"(median {r['rmse_median']:.3e}, O2 {r['rmse_o2']:.3e})"
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sweep_results.csv", index=False)
    with open(outdir / "gan_health.json", "w") as f:
        json.dump(gan_health, f, indent=2)

    # --- plot: RMSE vs number of observed points ---
    labels = {
        "linear": "Linear interp",
        "cubic_spline": "Cubic spline",
        "gp_rbf": "GP (RBF)",
        "gp_matern": "GP (Matern 5/2)",
        "cnn_supervised": "CNN (supervised)",
        "gan": "GAN (fixed)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for name, label in labels.items():
        sub = df[df.model == name]
        axes[0].plot(sub.n_lr, sub.rmse_mean, "o-", label=label)
        axes[1].plot(sub.n_lr, sub.rmse_o2, "o-", label=label)
    for ax, title in zip(axes, ["LJ + Morse test set (mean)", "O2 (out-of-family)"]):
        ax.set_yscale("log")
        ax.set_xlabel("observed low-res points")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMSE on 128-pt surface (normalized energy)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "sweep_rmse.png", dpi=150)

    # --- markdown summary ---
    piv = df.pivot(index="n_lr", columns="model", values="rmse_mean")
    piv_o2 = df.pivot(index="n_lr", columns="model", values="rmse_o2")
    with open(outdir / "SUMMARY.md", "w") as f:
        f.write("# Sweep: high-res surface RMSE vs number of observed points\n\n")
        f.write(f"High-res grid: {HR_SIZE} points; energies normalized to [-1, 1].\n")
        f.write(f"Point placement: {placement}.\n")
        f.write(f"Device: {device}; scale x{SCALE}.\n")
        f.write(f"Families: {', '.join(FAMILIES)}.\n")
        f.write(f"Train: {len(FAMILIES) * n_train_per_family} curves; ")
        f.write(f"test: {len(FAMILIES) * N_TEST_PER_FAMILY} curves.\n\n")
        f.write("## Mean RMSE, in-family test set\n\n")
        f.write(piv.to_markdown(floatfmt=".3e") + "\n\n")
        f.write("## RMSE, O2 Reudenberg curve (out-of-family)\n\n")
        f.write(piv_o2.to_markdown(floatfmt=".3e") + "\n\n")
        for fam in FAMILIES:
            piv_fam = df.pivot(index="n_lr", columns="model", values=f"rmse_{fam}")
            f.write(f"## Mean RMSE, {fam} test curves\n\n")
            f.write(piv_fam.to_markdown(floatfmt=".3e") + "\n\n")
        f.write("## GAN training health (tail averages)\n\n")
        f.write(pd.DataFrame(gan_health).T.to_markdown(floatfmt=".3f") + "\n\n")
        f.write(
            "Note: D(real)=0 / D(fake)=1 'accuracy' with adv loss ~0.80 means the\n"
            "discriminator outputs a constant p~0.45 for everything - the\n"
            "label-smoothed equilibrium where fakes are indistinguishable from real.\n"
        )
    print(f"\nWrote results to {outdir}")


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/sweep"),
        sys.argv[2] if len(sys.argv) > 2 else "uniform",
    )
