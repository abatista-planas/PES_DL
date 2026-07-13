import os

import numpy as np
import pytest

from pes_1D.anomaly import baselines, detector, mitbih  # type: ignore
from pes_1D.anomaly.benchmark import compute_metrics  # type: ignore
from pes_1D.anomaly.ecg_anomalies import corrupt_batch  # type: ignore


def _synthetic_beats(n, length=320, kind="normal", seed=0):
    """Smooth low-frequency 'normal' beats; noisy 'anomaly' beats."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)
    out = []
    for _ in range(n):
        base = np.exp(-((t - 0.5) ** 2) / (2 * 0.02)) * rng.uniform(0.8, 1.2)
        if kind == "normal":
            beat = base + 0.02 * rng.standard_normal(length)
        else:
            beat = base + 0.8 * rng.standard_normal(length)
        out.append(mitbih.znorm(beat).astype(np.float32))
    return np.stack(out)


# --------------------------------------------------------------------------- #
# preprocessing utilities (no download)
# --------------------------------------------------------------------------- #
def test_resample_beat_length_and_norm():
    raw = np.random.default_rng(0).standard_normal(288)
    beat = mitbih.resample_beat(raw, beat_len=320)
    assert beat.shape == (320,)
    assert abs(float(beat.mean())) < 1e-5
    assert abs(float(beat.std()) - 1.0) < 1e-2


def test_corrupt_batch_shape_and_change():
    X = _synthetic_beats(16, kind="normal")
    Xa = corrupt_batch(X, seed=1)
    assert Xa.shape == X.shape
    assert not np.allclose(Xa, X)  # deformations actually change the beats


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_compute_metrics_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95])
    m = compute_metrics(scores, y)
    assert m["AUROC"] == pytest.approx(1.0)
    assert m["AUPRC"] == pytest.approx(1.0)
    assert m["F1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# detector trains and produces valid scores
# --------------------------------------------------------------------------- #
def test_detector_trains_and_scores_in_range():
    Xn = _synthetic_beats(40, kind="normal", seed=1)
    Xa = _synthetic_beats(40, kind="anomaly", seed=2)
    X = np.concatenate([Xn, Xa])
    y = np.concatenate([np.zeros(40), np.ones(40)])
    model = detector.train_discriminator(
        X, y, grid_size=320, epochs=3, batch_size=32, device="cpu", seed=0
    )
    scores = detector.evaluate_scores(model, X, device="cpu")
    assert scores.shape == (80,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    # separable synthetic data: after a few epochs it should be better than chance
    assert compute_metrics(scores, y)["AUROC"] > 0.6


# --------------------------------------------------------------------------- #
# a PyOD baseline runs end-to-end
# --------------------------------------------------------------------------- #
def test_pyod_baseline_runs():
    Xn = _synthetic_beats(40, kind="normal", seed=1)
    Xte = np.concatenate(
        [_synthetic_beats(10, kind="normal", seed=3), _synthetic_beats(10, kind="anomaly", seed=4)]
    )
    scores = baselines.run_pyod_baselines(Xn, Xte, seed=0)
    assert "IForest" in scores
    assert scores["IForest"].shape == (20,)
    assert np.all(np.isfinite(scores["IForest"]))


# --------------------------------------------------------------------------- #
# real MIT-BIH extraction (slow; opt in with MITBIH_E2E=1)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("MITBIH_E2E") != "1",
    reason="set MITBIH_E2E=1 to run the MIT-BIH download/extraction test",
)
def test_extract_real_record(tmp_path):
    data_dir = mitbih.download(str(tmp_path))
    X, y = mitbih.extract_record(os.path.join(data_dir, "100"))
    assert X.shape[1] == 320
    assert len(X) == len(y)
    assert set(np.unique(y)).issubset({0, 1})
