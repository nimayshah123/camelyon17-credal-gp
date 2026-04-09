"""
credal_gp.py — Credal Gaussian Process with diverse kernel set.

Theory
------
A Credal GP is a set of GPs indexed by hyperparameter θ ∈ Θ:

    {GP(0, k_θ) : θ ∈ Θ}

Each member k_θ is an RBF kernel with a distinct (lengthscale, output_scale):

    k_θ(x, x') = σ² · exp(−0.5 · (x − x')² / l²)

The posterior predictive variance for member θ at test point x* is:

    σ²_θ(x*) = k_θ(x*, x*) − k_θ(x*, X)[k_θ(X, X) + ε·I]⁻¹ k_θ(X, x*)

The credal width at x* is:

    W(x*) = sup_θ σ²_θ(x*) − inf_θ σ²_θ(x*)

Because diverse kernels have very different lengthscales:
  - Where training data is dense, all kernels are forced to agree → narrow W.
  - Where training data is absent (OOD region), kernels disagree → wide W.

This gives automatic epistemic uncertainty widening in OOD regions without
any explicit OOD labelling.

The center prediction (mean label probability) is the average posterior mean
over all kernel members, passed through a sigmoid squashing for probabilities.

Classes
-------
RBFKernel   : single RBF kernel with configurable params.
CredalGP    : the full credal set with fit / predict / width APIs.
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ---------------------------------------------------------------------------
# RBF kernel
# ---------------------------------------------------------------------------
class RBFKernel:
    """
    Radial Basis Function (squared-exponential) kernel.

    k(x, x') = σ² · exp(−0.5 · (x − x')² / l²)

    Args
    ----
    lengthscale : float
        Controls how quickly correlations decay with distance.
    output_scale : float
        Overall variance amplitude (σ²).
    """

    def __init__(self, lengthscale: float, output_scale: float):
        self.l  = lengthscale
        self.s2 = output_scale

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix K[i,j] = k(X1[i], X2[j]).

        Args
        ----
        X1 : np.ndarray, shape (N,) or (N, 1)
        X2 : np.ndarray, shape (M,) or (M, 1)

        Returns
        -------
        K : np.ndarray, shape (N, M)
        """
        X1 = np.atleast_1d(X1).ravel()
        X2 = np.atleast_1d(X2).ravel()
        diff = X1[:, None] - X2[None, :]          # (N, M)
        return self.s2 * np.exp(-0.5 * diff**2 / self.l**2)

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Return diagonal k(X[i], X[i]) = σ² for all i."""
        return self.s2 * np.ones(len(np.atleast_1d(X).ravel()))


# ---------------------------------------------------------------------------
# Credal GP
# ---------------------------------------------------------------------------
class CredalGP:
    """
    Credal Gaussian Process: a set of GPs with diverse RBF kernels.

    The kernel set Θ is the Cartesian product of lengthscales × output_scales,
    giving |Θ| = len(lengthscales) × len(output_scales) kernels.

    Parameters
    ----------
    lengthscales : list of float
        RBF lengthscales forming the credal set.
        Default: config.CREDAL_LENGTHSCALES.
    output_scales : list of float
        Output variance values.
        Default: config.CREDAL_OUTPUT_SCALES.
    noise : float
        Observation noise added to the diagonal of K(X, X).
        Default: config.NOISE.
    subsample : int or None
        If set, randomly subsample training data to at most this many points
        (reduces O(n³) kernel inversion cost).
        Default: config.GP_SUBSAMPLE.
    """

    def __init__(
        self,
        lengthscales: List[float] = None,
        output_scales: List[float] = None,
        noise: float = None,
        subsample: int = None,
    ):
        self.lengthscales  = lengthscales  or config.CREDAL_LENGTHSCALES
        self.output_scales = output_scales or config.CREDAL_OUTPUT_SCALES
        self.noise         = noise   if noise    is not None else config.NOISE
        self.subsample     = subsample if subsample is not None \
                             else config.GP_SUBSAMPLE

        # Build the full kernel set as Cartesian product
        self.kernels: List[RBFKernel] = []
        for l in self.lengthscales:
            for s in self.output_scales:
                self.kernels.append(RBFKernel(lengthscale=l, output_scale=s))

        # Fitted quantities (populated by fit())
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self._alpha: List[np.ndarray]  = []   # (K_θ + ε I)^{-1} y per kernel
        self._L:     List[np.ndarray]  = []   # Cholesky factors per kernel
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'CredalGP':
        """
        Fit the credal GP to training data.

        Pre-computes Cholesky decompositions and alpha vectors for all
        kernel members so that predict() is fast.

        Args
        ----
        X_train : np.ndarray, shape (N,)
            1-D feature values (the top-expansion PCA feature).
        y_train : np.ndarray, shape (N,)
            Binary labels (0/1) or floats.

        Returns
        -------
        self
        """
        rng = np.random.RandomState(config.RANDOM_SEED)

        X = np.atleast_1d(X_train).ravel().astype(float)
        y = np.atleast_1d(y_train).ravel().astype(float)

        # Subsample to keep kernel inversion tractable
        if self.subsample and len(X) > self.subsample:
            idx = rng.choice(len(X), size=self.subsample, replace=False)
            X   = X[idx]
            y   = y[idx]

        self.X_train = X
        self.y_train = y
        n = len(X)

        self._alpha = []
        self._L     = []

        for kernel in self.kernels:
            K = kernel(X, X)                          # (n, n)
            K_noisy = K + self.noise * np.eye(n)     # add noise

            # Cholesky: K_noisy = L @ L.T
            try:
                L = np.linalg.cholesky(K_noisy)
            except np.linalg.LinAlgError:
                # Add jitter if not PSD
                K_noisy += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K_noisy)

            # alpha = (K + ε I)^{-1} y  via two triangular solves
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

            self._L.append(L)
            self._alpha.append(alpha)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Posterior mean for a single kernel
    # ------------------------------------------------------------------
    def _posterior_mean(
        self, X_test: np.ndarray, kernel_idx: int
    ) -> np.ndarray:
        """
        Compute posterior mean for one kernel member.

        Args
        ----
        X_test : np.ndarray, shape (M,)
        kernel_idx : int

        Returns
        -------
        mean : np.ndarray, shape (M,)
        """
        kernel = self.kernels[kernel_idx]
        K_star = kernel(X_test, self.X_train)          # (M, n)
        return K_star @ self._alpha[kernel_idx]

    # ------------------------------------------------------------------
    # Posterior variance for a single kernel
    # ------------------------------------------------------------------
    def posterior_variance(
        self, X_test: np.ndarray, kernel_idx: int
    ) -> np.ndarray:
        """
        Compute posterior predictive variance for one kernel member.

        σ²_θ(x*) = k(x*,x*) − k(x*,X)[K(X,X)+εI]^{-1} k(X,x*)

        Args
        ----
        X_test : np.ndarray, shape (M,)
        kernel_idx : int

        Returns
        -------
        var : np.ndarray, shape (M,)  (non-negative)
        """
        kernel = self.kernels[kernel_idx]
        X_test = np.atleast_1d(X_test).ravel()

        k_diag = kernel.diag(X_test)                   # (M,)
        K_star = kernel(X_test, self.X_train)          # (M, n)
        L      = self._L[kernel_idx]

        # v = L^{-1} k(X, x*)  →  via forward solve
        v = np.linalg.solve(L, K_star.T)               # (n, M)
        var = k_diag - np.sum(v**2, axis=0)            # (M,)
        return np.maximum(var, 0.0)

    # ------------------------------------------------------------------
    # Full credal prediction
    # ------------------------------------------------------------------
    def predict(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Credal GP prediction at test points.

        Returns the center prediction (mean over all kernel members) and
        the credal envelope (pointwise inf and sup of posterior means), plus
        the credal width = sup σ²_θ − inf σ²_θ.

        Args
        ----
        X_test : np.ndarray, shape (M,)
            Test feature values along the top-expansion axis.

        Returns
        -------
        center_mean   : np.ndarray, shape (M,)
            Average posterior mean over all kernel members (raw GP output).
        lower_mean    : np.ndarray, shape (M,)
            Pointwise infimum of posterior means.
        upper_mean    : np.ndarray, shape (M,)
            Pointwise supremum of posterior means.
        credal_width  : np.ndarray, shape (M,)
            sup_θ σ²_θ(x*) − inf_θ σ²_θ(x*).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        X_test = np.atleast_1d(X_test).ravel().astype(float)

        all_means = []
        all_vars  = []

        for k_idx in range(len(self.kernels)):
            m = self._posterior_mean(X_test, k_idx)
            v = self.posterior_variance(X_test, k_idx)
            all_means.append(m)
            all_vars.append(v)

        all_means = np.stack(all_means, axis=0)  # (K, M)
        all_vars  = np.stack(all_vars,  axis=0)  # (K, M)

        center_mean  = np.mean(all_means, axis=0)
        lower_mean   = np.min(all_means,  axis=0)
        upper_mean   = np.max(all_means,  axis=0)
        credal_width = np.max(all_vars, axis=0) - np.min(all_vars, axis=0)

        return center_mean, lower_mean, upper_mean, credal_width

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return calibrated probability estimates via sigmoid of center mean.

        Args
        ----
        X_test : np.ndarray, shape (M,)

        Returns
        -------
        proba : np.ndarray, shape (M,)  values in [0, 1].
        """
        center_mean, _, _, _ = self.predict(X_test)
        return _sigmoid(center_mean)

    def credal_width_at(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return only the credal width (convenience wrapper).

        Args
        ----
        X_test : np.ndarray, shape (M,)

        Returns
        -------
        credal_width : np.ndarray, shape (M,)
        """
        _, _, _, width = self.predict(X_test)
        return width

    # ------------------------------------------------------------------
    # Single-kernel GP for baseline comparison
    # ------------------------------------------------------------------
    def single_gp_variance(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return posterior variance of the middle (median) kernel only.

        Used to contrast with credal width in Figure 2.

        Args
        ----
        X_test : np.ndarray, shape (M,)

        Returns
        -------
        var : np.ndarray, shape (M,)
        """
        mid_idx = len(self.kernels) // 2
        return self.posterior_variance(X_test, mid_idx)

    def single_gp_mean(self, X_test: np.ndarray) -> np.ndarray:
        """
        Return posterior mean of the middle kernel.

        Args
        ----
        X_test : np.ndarray, shape (M,)

        Returns
        -------
        mean : np.ndarray, shape (M,)
        """
        mid_idx = len(self.kernels) // 2
        return self._posterior_mean(X_test, mid_idx)

    # ------------------------------------------------------------------
    # Credal width after adding a hypothetical new domain
    # ------------------------------------------------------------------
    def expected_width_with_new_domain(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        X_query: np.ndarray,
    ) -> np.ndarray:
        """
        Compute credal width if we augment training data with a new hospital.

        Used by domain_selection.py to rank hospitals by how much their
        addition would narrow the credal width at query points.

        Args
        ----
        X_new   : np.ndarray, shape (N_new,)
            Feature values of the candidate new hospital.
        y_new   : np.ndarray, shape (N_new,)
            Labels for the new hospital.
        X_query : np.ndarray, shape (M,)
            Query points at which to evaluate the resulting width.

        Returns
        -------
        width_after : np.ndarray, shape (M,)
        """
        # Augment training set
        X_aug = np.concatenate([self.X_train, X_new])
        y_aug = np.concatenate([self.y_train, y_new])

        # Build temporary GP with same kernel set
        tmp_gp = CredalGP(
            lengthscales=self.lengthscales,
            output_scales=self.output_scales,
            noise=self.noise,
            subsample=None,  # keep all augmented data
        )
        tmp_gp.fit(X_aug, y_aug)
        _, _, _, width_after = tmp_gp.predict(X_query)
        return width_after

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_kernels = len(self.kernels)
        ls_str    = ", ".join(f"{l}" for l in self.lengthscales)
        os_str    = ", ".join(f"{s}" for s in self.output_scales)
        status    = "fitted" if self._fitted else "not fitted"
        return (
            f"CredalGP({status}, {n_kernels} kernels, "
            f"lengthscales=[{ls_str}], output_scales=[{os_str}], "
            f"noise={self.noise})"
        )


# ---------------------------------------------------------------------------
# Convenience: fit a credal GP on the top-expansion feature
# ---------------------------------------------------------------------------
def fit_credal_gp_on_top_feature(
    features_dict: Dict[int, np.ndarray],
    labels_dict: Dict[int, np.ndarray],
    top_feature_idx: int,
    hospital_ids: Optional[List[int]] = None,
) -> CredalGP:
    """
    Fit a CredalGP using one PCA feature as the 1-D input.

    Args
    ----
    features_dict   : {hospital_id: np.ndarray (N, D)}
    labels_dict     : {hospital_id: np.ndarray (N,)}
    top_feature_idx : int — which PCA component to use as x-axis.
    hospital_ids    : list of int — which hospitals to train on.
                      Defaults to config.SOURCE_HOSPITALS.

    Returns
    -------
    gp : fitted CredalGP instance.
    """
    if hospital_ids is None:
        hospital_ids = config.SOURCE_HOSPITALS

    X_parts, y_parts = [], []
    for h in hospital_ids:
        if h not in features_dict:
            continue
        X_parts.append(features_dict[h][:, top_feature_idx])
        y_parts.append(labels_dict[h].astype(float))

    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)

    gp = CredalGP()
    gp.fit(X_train, y_train)
    return gp
