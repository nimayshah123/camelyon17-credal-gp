"""
baselines.py — Deterministic and probabilistic baselines for comparison.

Three baselines are implemented, all using scikit-learn on PCA features
(no deep learning needed after feature extraction):

1. ERMBaseline
   Empirical Risk Minimisation via LogisticRegression on PCA features.
   Represents the standard deterministic approach.

2. MCDropoutBaseline
   Approximates MC Dropout by training N_MC_SAMPLES MLPClassifiers with
   different random seeds and averaging their predictions.  Each run sees
   random dropout-like variation from initialisation differences.
   Returns mean probability + std as uncertainty.

3. DeepEnsembleBaseline
   Trains N_ENSEMBLE MLPClassifiers with different random seeds.
   Returns mean probability + std as uncertainty.

All classes follow a consistent interface:
    fit(X_train, y_train)
    predict_proba(X_test) -> (mean_proba, uncertainty)

Where uncertainty is None for ERM (point estimate) and a std array for the
stochastic methods.
"""

import os
import sys
import time
import numpy as np
from typing import Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _fmt(seconds: float) -> str:
    """Format elapsed seconds as m:ss or s."""
    if seconds >= 60:
        return f"{int(seconds//60)}m {int(seconds%60):02d}s"
    return f"{seconds:.1f}s"


# ---------------------------------------------------------------------------
# ERM Baseline
# ---------------------------------------------------------------------------
class ERMBaseline:
    """
    Empirical Risk Minimisation: LogisticRegression on PCA features.

    This represents the standard (non-probabilistic) baseline.  It uses
    L2-regularised logistic regression which is a well-calibrated point
    estimator on in-distribution data but has no notion of epistemic
    uncertainty.

    Args
    ----
    C : float
        Inverse regularisation strength.  Larger → less regularisation.
    max_iter : int
        Maximum iterations for solver convergence.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C        = C
        self.max_iter = max_iter
        self._pipe: Optional[Pipeline] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ERMBaseline':
        """
        Fit the ERM model.

        Args
        ----
        X : np.ndarray, shape (N, D)
        y : np.ndarray, shape (N,)  — binary labels.

        Returns
        -------
        self
        """
        self._pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=config.RANDOM_SEED,
                solver='lbfgs',
            )),
        ])
        self._pipe.fit(X, y)
        return self

    def predict_proba(
        self, X: np.ndarray
    ) -> np.ndarray:
        """
        Return class probabilities (no uncertainty).

        Args
        ----
        X : np.ndarray, shape (M, D)

        Returns
        -------
        proba : np.ndarray, shape (M,)  — P(tumor | x)
        """
        if self._pipe is None:
            raise RuntimeError("Call fit() first.")
        proba = self._pipe.predict_proba(X)[:, 1]
        return proba

    def name(self) -> str:
        return "ERM (LogisticRegression)"


# ---------------------------------------------------------------------------
# MC Dropout Baseline
# ---------------------------------------------------------------------------
class MCDropoutBaseline:
    """
    MC Dropout approximation via an ensemble of MLP classifiers.

    True MC Dropout requires a neural network with dropout layers.  Here we
    simulate the distributional effect by training N_MC_SAMPLES shallow MLPs
    with different random seeds.  Each seed induces a different random
    initialisation, giving diverse predictions analogous to stochastic forward
    passes.

    This is a faithful proxy when we do not have access to a full deep network
    in this pipeline (all deep features are pre-computed).

    Args
    ----
    n_samples : int
        Number of forward-pass samples (= number of MLPs trained).
    hidden_layer_sizes : tuple
        MLP hidden layer sizes.
    dropout_rate : float
        Conceptual dropout rate (controls MLP alpha regularisation proxy).
    """

    def __init__(
        self,
        n_samples: int = None,
        hidden_layer_sizes: tuple = (128, 64),
        dropout_rate: float = 0.3,
    ):
        self.n_samples          = n_samples or config.N_MC_SAMPLES
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate       = dropout_rate
        self._models: list      = []
        self._scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MCDropoutBaseline':
        """
        Train N MC-Dropout MLP models.

        Args
        ----
        X : np.ndarray, shape (N, D)
        y : np.ndarray, shape (N,)

        Returns
        -------
        self
        """
        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X)

        self._models = []
        t0_all = time.time()
        for i in range(self.n_samples):
            rng_seed = config.RANDOM_SEED + i
            alpha    = self.dropout_rate * (0.8 + 0.4 * np.random.RandomState(rng_seed).rand())

            t0 = time.time()
            print(f"    MC Dropout [{i+1}/{self.n_samples}] fitting …", end=" ", flush=True)
            clf = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation='relu',
                alpha=alpha,
                max_iter=200,
                random_state=rng_seed,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
            )
            clf.fit(X_scaled, y)
            self._models.append(clf)
            elapsed = time.time() - t0
            total_so_far = time.time() - t0_all
            remaining = (total_so_far / (i + 1)) * (self.n_samples - i - 1)
            print(f"done ({_fmt(elapsed)}) — est. remaining: {_fmt(remaining)}")

        return self

    def predict_proba(
        self, X: np.ndarray, n_samples: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return mean and uncertainty from N stochastic forward passes.

        Args
        ----
        X        : np.ndarray, shape (M, D)
        n_samples: int, optional — how many models to use (≤ self.n_samples).

        Returns
        -------
        mean_proba  : np.ndarray, shape (M,)  — mean P(tumor | x)
        uncertainty : np.ndarray, shape (M,)  — std across MC samples
        """
        if not self._models:
            raise RuntimeError("Call fit() first.")

        n_use   = n_samples or len(self._models)
        X_scaled = self._scaler.transform(X)

        all_probas = np.stack(
            [m.predict_proba(X_scaled)[:, 1] for m in self._models[:n_use]],
            axis=0
        )  # (n_samples, M)

        mean_proba  = all_probas.mean(axis=0)
        uncertainty = all_probas.std(axis=0)
        return mean_proba, uncertainty

    def name(self) -> str:
        return f"MC Dropout (N={self.n_samples})"


# ---------------------------------------------------------------------------
# Deep Ensemble Baseline
# ---------------------------------------------------------------------------
class DeepEnsembleBaseline:
    """
    Deep Ensemble: multiple independently trained MLP classifiers.

    Each ensemble member is trained with a different random seed, giving
    diverse predictions from different local minima.  Uncertainty = std of
    ensemble predictions at a point.

    Args
    ----
    n_ensemble : int
        Number of ensemble members.
    hidden_layer_sizes : tuple
        MLP hidden layer sizes.
    """

    def __init__(
        self,
        n_ensemble: int = None,
        hidden_layer_sizes: tuple = (256, 128, 64),
    ):
        self.n_ensemble         = n_ensemble or config.N_ENSEMBLE
        self.hidden_layer_sizes = hidden_layer_sizes
        self._models: list      = []
        self._scaler: Optional[StandardScaler] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DeepEnsembleBaseline':
        """
        Train all ensemble members independently.

        Args
        ----
        X : np.ndarray, shape (N, D)
        y : np.ndarray, shape (N,)

        Returns
        -------
        self
        """
        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X)

        self._models = []
        t0_all = time.time()
        for i in range(self.n_ensemble):
            t0 = time.time()
            print(f"    Ensemble member [{i+1}/{self.n_ensemble}] fitting …", end=" ", flush=True)
            clf = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation='relu',
                alpha=1e-4,
                max_iter=300,
                random_state=config.RANDOM_SEED + i * 100,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
            )
            clf.fit(X_scaled, y)
            self._models.append(clf)
            elapsed = time.time() - t0
            total_so_far = time.time() - t0_all
            remaining = (total_so_far / (i + 1)) * (self.n_ensemble - i - 1)
            print(f"done ({_fmt(elapsed)}) — est. remaining: {_fmt(remaining)}")

        return self

    def predict_proba(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return ensemble mean and std at test points.

        Args
        ----
        X : np.ndarray, shape (M, D)

        Returns
        -------
        mean_proba  : np.ndarray, shape (M,)
        uncertainty : np.ndarray, shape (M,)  — std across ensemble members
        """
        if not self._models:
            raise RuntimeError("Call fit() first.")

        X_scaled = self._scaler.transform(X)

        all_probas = np.stack(
            [m.predict_proba(X_scaled)[:, 1] for m in self._models],
            axis=0
        )  # (n_ensemble, M)

        mean_proba  = all_probas.mean(axis=0)
        uncertainty = all_probas.std(axis=0)
        return mean_proba, uncertainty

    def name(self) -> str:
        return f"Deep Ensemble (N={self.n_ensemble})"


# ---------------------------------------------------------------------------
# Convenience: train all baselines on source data
# ---------------------------------------------------------------------------
def train_all_baselines(
    features_dict: dict,
    labels_dict: dict,
    hospital_ids: list = None,
    verbose: bool = True,
) -> dict:
    """
    Train ERM, MC Dropout, and Deep Ensemble on source hospital data.

    Args
    ----
    features_dict : {hospital_id: np.ndarray (N, D)}
    labels_dict   : {hospital_id: np.ndarray (N,)}
    hospital_ids  : which hospitals to include in training.
                    Defaults to config.SOURCE_HOSPITALS.
    verbose : bool

    Returns
    -------
    dict : {'erm': ERMBaseline, 'mc': MCDropoutBaseline,
            'ensemble': DeepEnsembleBaseline}
    """
    if hospital_ids is None:
        hospital_ids = config.SOURCE_HOSPITALS

    X_parts, y_parts = [], []
    for h in hospital_ids:
        if h in features_dict:
            X_parts.append(features_dict[h])
            y_parts.append(labels_dict[h])

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)

    # Optional training sample cap (config.MAX_TRAIN_SAMPLES = None means no cap)
    cap = config.MAX_TRAIN_SAMPLES
    if cap is not None and len(X_train) > cap:
        rng = np.random.RandomState(config.RANDOM_SEED)
        idx = rng.choice(len(X_train), size=cap, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        if verbose:
            print(f"  Subsampled to {cap:,} training samples")

    if verbose:
        print(f"  Training on {len(X_train):,} samples from hospitals "
              f"{hospital_ids}")

    results = {}

    if verbose:
        print("  Training ERM baseline …")
    erm = ERMBaseline()
    erm.fit(X_train, y_train)
    results['erm'] = erm

    if verbose:
        print("  Training MC Dropout baseline …")
    mc = MCDropoutBaseline()
    mc.fit(X_train, y_train)
    results['mc'] = mc

    if verbose:
        print("  Training Deep Ensemble baseline …")
    ens = DeepEnsembleBaseline()
    ens.fit(X_train, y_train)
    results['ensemble'] = ens

    if verbose:
        print("  All baselines trained.")

    return results
