"""
expansion.py — Expansion function computation for PCA features.

The expansion function e_φ(z) measures how much a given feature z shifts
across hospital domains:

    e_φ(z) = σ²_between(z) / σ²_within(z)

where
    σ²_between = variance of per-domain means  (between-domain spread)
    σ²_within  = mean of per-domain variances  (average within-domain spread)

A high expansion value means the feature is a strong "domain shift" indicator:
its distribution moves substantially from hospital to hospital relative to
how spread out it is within each hospital.

Functions
---------
compute_expansion         : scalar expansion for one feature index.
compute_all_expansions    : expansion for all PCA dims.
get_top_expansion_features: sorted list of (feature_idx, expansion_value).
estimate_expansion_from_source_domains : same, restricted to source hospitals.
per_hospital_feature_means: helper returning mean feature value per hospital.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_expansion(
    features_dict: Dict[int, np.ndarray],
    feature_idx: int,
    hospital_ids: Optional[List[int]] = None,
) -> float:
    """
    Compute the expansion function value for a single PCA feature.

    Args
    ----
    features_dict : dict
        {hospital_id: np.ndarray shape (N_h, D)} — PCA features per hospital.
    feature_idx : int
        Which PCA dimension to evaluate (0 ≤ feature_idx < D).
    hospital_ids : list of int, optional
        Subset of hospitals to use.  Defaults to all keys in features_dict.

    Returns
    -------
    float
        e_φ(z) = var(per_domain_means) / mean(per_domain_variances).
        Returns np.nan if any hospital has fewer than 2 samples.
    """
    if hospital_ids is None:
        hospital_ids = sorted(features_dict.keys())

    per_domain_means = []
    per_domain_vars  = []

    for h in hospital_ids:
        feat = features_dict[h][:, feature_idx]
        if len(feat) < 2:
            return np.nan
        per_domain_means.append(np.mean(feat))
        per_domain_vars.append(np.var(feat, ddof=1))

    sigma2_between = np.var(per_domain_means, ddof=1)
    sigma2_within  = np.mean(per_domain_vars)

    if sigma2_within < 1e-12:
        return np.nan

    return float(sigma2_between / sigma2_within)


def compute_all_expansions(
    features_dict: Dict[int, np.ndarray],
    hospital_ids: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Compute expansion values for every PCA feature dimension.

    Args
    ----
    features_dict : dict
        {hospital_id: np.ndarray shape (N_h, D)}.
    hospital_ids : list of int, optional
        Restrict computation to these hospitals.

    Returns
    -------
    dict : {feature_idx: expansion_value (float)}
    """
    if hospital_ids is None:
        hospital_ids = sorted(features_dict.keys())

    sample_mat = next(iter(features_dict.values()))
    n_features = sample_mat.shape[1]

    expansions = {}
    for f in range(n_features):
        expansions[f] = compute_expansion(features_dict, f, hospital_ids)

    return expansions


def get_top_expansion_features(
    features_dict: Dict[int, np.ndarray],
    n: int = 5,
    hospital_ids: Optional[List[int]] = None,
) -> List[Tuple[int, float]]:
    """
    Return the n features with the highest expansion values.

    Args
    ----
    features_dict : dict
        {hospital_id: np.ndarray shape (N_h, D)}.
    n : int
        Number of top features to return.
    hospital_ids : list of int, optional
        Restrict to these hospitals.

    Returns
    -------
    list of (feature_idx, expansion_value) sorted descending by expansion.
    """
    expansions = compute_all_expansions(features_dict, hospital_ids)
    # Remove nan values
    valid = {k: v for k, v in expansions.items() if not np.isnan(v)}
    sorted_feats = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    return sorted_feats[:n]


def estimate_expansion_from_source_domains(
    features_dict: Dict[int, np.ndarray],
    n: int = None,
) -> Dict[int, float]:
    """
    Compute expansion values using only source (non-held-out) hospitals.

    This simulates the realistic scenario where the practitioner only has
    access to source hospitals when computing the expansion function and
    deciding which hospital to collect next.

    Args
    ----
    features_dict : dict
        {hospital_id: np.ndarray shape (N_h, D)} — all 5 hospitals.
    n : int, optional
        If given, return only top-n features.  Otherwise return all.

    Returns
    -------
    dict : {feature_idx: expansion_value} using source hospitals only.
    """
    src_ids = config.SOURCE_HOSPITALS
    # Filter to only source hospitals that actually exist in features_dict
    src_ids = [h for h in src_ids if h in features_dict]

    expansions = compute_all_expansions(features_dict, hospital_ids=src_ids)
    valid = {k: v for k, v in expansions.items() if not np.isnan(v)}

    if n is not None:
        sorted_feats = sorted(valid.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_feats[:n])

    return valid


def per_hospital_feature_means(
    features_dict: Dict[int, np.ndarray],
    feature_indices: List[int],
    hospital_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute per-hospital mean values for a set of features.

    Args
    ----
    features_dict : dict
        {hospital_id: np.ndarray shape (N_h, D)}.
    feature_indices : list of int
        Which PCA dimensions to include.
    hospital_ids : list of int, optional
        Which hospitals to include (rows).

    Returns
    -------
    means : np.ndarray, shape (n_hospitals, n_features)
        means[i, j] = mean of feature_indices[j] for hospital_ids[i].
    """
    if hospital_ids is None:
        hospital_ids = sorted(features_dict.keys())

    means = np.zeros((len(hospital_ids), len(feature_indices)))
    for row, h in enumerate(hospital_ids):
        for col, f in enumerate(feature_indices):
            means[row, col] = np.mean(features_dict[h][:, f])

    return means


def compute_ood_distance(
    features_dict: Dict[int, np.ndarray],
    ood_hospital: int = None,
    feature_idx: int = None,
) -> np.ndarray:
    """
    Compute how far each OOD test point is from the source-domain coverage.

    Distance is measured as the absolute deviation from the nearest source
    domain mean along the given feature axis.  Used to validate that credal
    width correlates with OOD distance.

    Args
    ----
    features_dict : dict
        {hospital_id: np.ndarray shape (N_h, D)}.
    ood_hospital : int, optional
        Defaults to config.HELD_OUT_HOSPITAL.
    feature_idx : int, optional
        Which feature to use.  Defaults to the top-expansion feature.

    Returns
    -------
    distances : np.ndarray, shape (N_ood,)
        Per-sample OOD distance along the chosen feature.
    """
    ood_hospital = ood_hospital if ood_hospital is not None \
                   else config.HELD_OUT_HOSPITAL

    if feature_idx is None:
        top_feats = get_top_expansion_features(features_dict, n=1,
                                               hospital_ids=config.SOURCE_HOSPITALS)
        feature_idx = top_feats[0][0]

    src_ids = [h for h in config.SOURCE_HOSPITALS if h in features_dict]
    src_means = np.array([
        np.mean(features_dict[h][:, feature_idx]) for h in src_ids
    ])

    ood_vals = features_dict[ood_hospital][:, feature_idx]

    # Distance to nearest source mean
    dists = np.array([
        np.min(np.abs(v - src_means)) for v in ood_vals
    ])
    return dists
