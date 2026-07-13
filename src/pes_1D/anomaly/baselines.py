"""PyOD anomaly-detection baselines, fit on normal beats only (train-on-normal).

Each detector is fit on the normal training beats and scored on the test set via
``decision_function`` (higher = more anomalous). Detectors are constructed lazily and
wrapped so that a single failing model (version/API drift) does not abort the run.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def _flatten(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(len(X), -1)


def _constructors(n_features: int, seed: int) -> Dict[str, Callable]:
    """Lazy constructors so import failures don't break the whole module."""
    from pyod.models.ecod import ECOD  # type: ignore
    from pyod.models.iforest import IForest  # type: ignore
    from pyod.models.lof import LOF  # type: ignore
    from pyod.models.ocsvm import OCSVM  # type: ignore
    from pyod.models.pca import PCA  # type: ignore

    ctors: Dict[str, Callable] = {
        "IForest": lambda: IForest(random_state=seed),
        "OCSVM": lambda: OCSVM(),
        "LOF": lambda: LOF(novelty=True),
        "ECOD": lambda: ECOD(),
        "PCA": lambda: PCA(random_state=seed),
    }

    def _auto_encoder():
        from pyod.models.auto_encoder import AutoEncoder  # type: ignore

        return AutoEncoder(epoch_num=20, verbose=0, random_state=seed)

    def _deep_svdd():
        from pyod.models.deep_svdd import DeepSVDD  # type: ignore

        return DeepSVDD(
            n_features=n_features, epochs=20, verbose=0, random_state=seed
        )

    ctors["AutoEncoder"] = _auto_encoder
    ctors["DeepSVDD"] = _deep_svdd
    return ctors


def run_pyod_baselines(
    X_train_normal: np.ndarray, X_test: np.ndarray, seed: int = 0, max_fit: int = 20000
) -> Dict[str, np.ndarray]:
    """Fit each baseline on normal beats; return {name: test anomaly scores}.

    ``max_fit`` caps the fit-set size (random subsample) so O(n^2) detectors
    (OCSVM, LOF) stay tractable on large beat sets.
    """
    Xtr = _flatten(X_train_normal)
    if max_fit and len(Xtr) > max_fit:
        rng = np.random.default_rng(seed)
        Xtr = Xtr[rng.choice(len(Xtr), size=max_fit, replace=False)]
    Xte = _flatten(X_test)
    results: Dict[str, np.ndarray] = {}
    for name, ctor in _constructors(Xtr.shape[1], seed).items():
        try:
            model = ctor()
            model.fit(Xtr)
            results[name] = np.asarray(model.decision_function(Xte), dtype=np.float64)
        except Exception as exc:  # keep the benchmark alive if one detector fails
            print(f"[baselines] {name} failed: {type(exc).__name__}: {exc}")
    return results
