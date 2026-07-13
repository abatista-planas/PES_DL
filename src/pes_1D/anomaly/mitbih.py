"""MIT-BIH arrhythmia data pipeline: download, beat segmentation, labeling.

Beats are extracted around annotated R-peaks, resampled to a fixed length, and
z-normalized. Labels follow the AAMI grouping: the N class (normal) is label 0, and
the S/V/F/Q classes (supraventricular, ventricular, fusion, unknown) are label 1.

Train/test use the standard patient-independent de Chazal split (DS1 = train,
DS2 = test); the four paced records (102, 104, 107, 217) are excluded, as is
conventional. Splitting by record avoids beat-level leakage between train and test.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from scipy.signal import resample  # type: ignore

# de Chazal et al. patient-independent split (44 records; paced records excluded).
MITBIH_DS1 = [
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
    201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
]
MITBIH_DS2 = [
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
    213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
]
PACED_EXCLUDED = [102, 104, 107, 217]

# AAMI beat-class grouping.
NORMAL_SYMBOLS = {"N", "L", "R", "e", "j"}
ABNORMAL_SYMBOLS = {"A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"}
BEAT_SYMBOLS = NORMAL_SYMBOLS | ABNORMAL_SYMBOLS


def download(data_dir: str) -> str:
    """Download the MIT-BIH arrhythmia database (mitdb) if not already present."""
    import wfdb  # type: ignore

    os.makedirs(data_dir, exist_ok=True)
    # a record is present iff its .hea header exists
    if not os.path.exists(os.path.join(data_dir, "100.hea")):
        wfdb.dl_database("mitdb", data_dir)
    return data_dir


def znorm(beat: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalize a single beat."""
    beat = np.asarray(beat, dtype=np.float64)
    return (beat - beat.mean()) / (beat.std() + 1e-8)


def resample_beat(window: np.ndarray, beat_len: int) -> np.ndarray:
    """Resample a raw beat window to ``beat_len`` samples and z-normalize."""
    return znorm(resample(np.asarray(window, dtype=np.float64), beat_len)).astype(
        np.float32
    )


def _mlii_channel(sig_name) -> int:
    """Index of the MLII lead if present, else channel 0."""
    for cand in ("MLII", "II", "ML2"):
        if cand in sig_name:
            return sig_name.index(cand)
    return 0


def extract_record(
    record_path: str, half_window: int = 144, beat_len: int = 320
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract labeled beats from a single record.

    Returns ``(X, y)`` with ``X`` shape ``(n_beats, beat_len)`` and ``y`` in {0, 1}.
    ``half_window`` samples are taken on each side of the R-peak (MIT-BIH fs = 360 Hz,
    so 144 ~= 0.4 s); the window is then resampled to ``beat_len``.
    """
    import wfdb  # type: ignore

    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")
    ch = _mlii_channel(list(record.sig_name))
    sig = record.p_signal[:, ch]
    n = sig.shape[0]

    beats, labels = [], []
    for sample, symbol in zip(ann.sample, ann.symbol):
        if symbol not in BEAT_SYMBOLS:
            continue
        lo, hi = sample - half_window, sample + half_window
        if lo < 0 or hi > n:
            continue
        beats.append(resample_beat(sig[lo:hi], beat_len))
        labels.append(0 if symbol in NORMAL_SYMBOLS else 1)

    if not beats:
        return np.empty((0, beat_len), np.float32), np.empty((0,), np.int64)
    return np.stack(beats), np.asarray(labels, dtype=np.int64)


def load_beats(
    records, data_dir: str, half_window: int = 144, beat_len: int = 320
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate labeled beats from a list of record numbers."""
    X_all, y_all = [], []
    for rec in records:
        path = os.path.join(data_dir, str(rec))
        X, y = extract_record(path, half_window, beat_len)
        if len(X):
            X_all.append(X)
            y_all.append(y)
    if not X_all:
        return np.empty((0, beat_len), np.float32), np.empty((0,), np.int64)
    return np.concatenate(X_all), np.concatenate(y_all)


def load_ds1_ds2(
    data_dir: str,
    half_window: int = 144,
    beat_len: int = 320,
    max_records: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the patient-independent DS1 (train) / DS2 (test) split.

    ``max_records`` (per split) allows a fast subset for quick runs.
    """
    ds1 = MITBIH_DS1[:max_records] if max_records else MITBIH_DS1
    ds2 = MITBIH_DS2[:max_records] if max_records else MITBIH_DS2
    X_train, y_train = load_beats(ds1, data_dir, half_window, beat_len)
    X_test, y_test = load_beats(ds2, data_dir, half_window, beat_len)
    return X_train, y_train, X_test, y_test


def load_beat_level(
    data_dir: str,
    half_window: int = 144,
    beat_len: int = 320,
    test_size: float = 0.5,
    seed: int = 0,
    max_records: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all beats and do a stratified beat-level random split.

    This is the easier (patient-dependent) setup comparable to BeatGAN's, as opposed
    to the patient-independent DS1/DS2 split.
    """
    from sklearn.model_selection import train_test_split  # type: ignore

    records = MITBIH_DS1 + MITBIH_DS2
    if max_records:
        records = records[:max_records]
    X, y = load_beats(records, data_dir, half_window, beat_len)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, y_train, X_test, y_test
