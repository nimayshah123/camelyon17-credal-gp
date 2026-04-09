"""
data_loader.py — Load the Camelyon17-WILDS dataset and split by hospital.

The Camelyon17-WILDS dataset contains 455,954 pathology image patches
(96x96 px) from 5 Dutch hospitals.  Each sample carries a metadata field
'hospital' (integer 0–4) that defines the natural domain structure arising
from different scanners and staining protocols.

This module returns raw PIL images + labels partitioned by hospital so that
downstream modules (feature_extractor) can iterate over them in batches.

Fast-dev mode
-------------
Set config.FAST_DEV = True to use only 500 randomly sampled patches per
hospital.  This lets the full pipeline run in minutes on CPU for testing.
"""

import os
import sys
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Allow imports from project root when running standalone
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_camelyon17(root: str = None, fast_dev: bool = None):
    """
    Load the Camelyon17-WILDS dataset and partition samples by hospital.

    Args
    ----
    root : str, optional
        Directory where WILDS will store / look for the dataset.
        Defaults to config.DATASET_ROOT.
    fast_dev : bool, optional
        If True, subsample to config.FAST_DEV_N patches per hospital.
        Defaults to config.FAST_DEV.

    Returns
    -------
    splits : dict
        {hospital_id (int): {'images': list[PIL.Image], 'y': np.ndarray(int)}}
        For all 5 hospitals (0–4).
    dataset_obj : wilds.datasets.camelyon17_dataset.Camelyon17Dataset
        The underlying WILDS dataset object (useful for metadata inspection).
    """
    try:
        from wilds import get_dataset
    except ImportError:
        raise ImportError(
            "WILDS is not installed.  Run: pip install wilds"
        )

    root      = root     or config.DATASET_ROOT
    fast_dev  = fast_dev if fast_dev is not None else config.FAST_DEV
    seed      = config.RANDOM_SEED

    print("  Loading Camelyon17-WILDS dataset …")
    dataset = get_dataset(
        dataset="camelyon17",
        root_dir=root,
        download=True,
    )

    # In WILDS Camelyon17 the 5 hospitals are spread across splits:
    #   train   → hospitals 0, 3, 4
    #   val     → hospital  1  (OOD val)
    #   test    → hospital  2  (OOD test)
    #   id_val  → hospitals 0, 3, 4  (in-distribution val — skip to avoid duplicates)
    # We load train + val + test to cover all 5 hospitals.
    subsets_to_load = ['train', 'val', 'test']

    # Accumulate per-hospital (global_idx, label, subset_ref) triples.
    # We store (local_index_within_subset, subset_ref) so we can retrieve images.
    per_hospital = defaultdict(lambda: {'local_indices': [], 'y': [], 'subset_ref': None})

    for split_name in subsets_to_load:
        subset = dataset.get_subset(split_name)
        metadata  = subset.metadata_array          # tensor
        hospitals = metadata[:, 0].numpy().astype(int)
        labels    = subset.y_array.numpy().astype(int)

        for local_idx in range(len(labels)):
            h = hospitals[local_idx]
            per_hospital[h]['local_indices'].append(local_idx)
            per_hospital[h]['y'].append(labels[local_idx])
            if per_hospital[h]['subset_ref'] is None:
                per_hospital[h]['subset_ref'] = subset

    n_total = sum(len(v['local_indices']) for v in per_hospital.values())
    print(f"  Total patches across all splits: {n_total:,}")

    rng = np.random.RandomState(seed)

    # Convert to arrays; optionally subsample
    result = {}
    for h in range(config.N_HOSPITALS):
        if h not in per_hospital:
            raise RuntimeError(
                f"Hospital {h} not found in any split of Camelyon17-WILDS."
            )
        idxs = np.array(per_hospital[h]['local_indices'])
        ys   = np.array(per_hospital[h]['y'])
        subset_ref = per_hospital[h]['subset_ref']

        if fast_dev and len(idxs) > config.FAST_DEV_N:
            chosen = rng.choice(len(idxs), size=config.FAST_DEV_N, replace=False)
            idxs   = idxs[chosen]
            ys     = ys[chosen]

        result[h] = {
            'subset_indices': idxs,   # local indices within subset_ref
            'y': ys,
            'dataset_ref': subset_ref,
        }
        label_str = f"  Hospital {h} ({config.HOSPITAL_NAMES[h]}): " \
                    f"{len(idxs):,} patches, " \
                    f"{ys.mean()*100:.1f}% tumor"
        print(label_str)

    return result, dataset


# Alias so external code can import via either name
load_camelyon17_by_hospital = load_camelyon17


def get_images_for_hospital(hospital_dict: dict, hospital_id: int):
    """
    Iterate over (PIL image, label) pairs for a given hospital.

    This is a generator to avoid loading all images into RAM at once.

    Args
    ----
    hospital_dict : dict
        Output of load_camelyon17(), keyed by hospital_id.
    hospital_id : int
        Which hospital to iterate over (0–4).

    Yields
    ------
    (PIL.Image, int) tuples.
    """
    h_data     = hospital_dict[hospital_id]
    subset     = h_data['dataset_ref']
    idxs       = h_data['subset_indices']
    ys         = h_data['y']

    for pos, (dataset_idx, label) in enumerate(zip(idxs, ys)):
        x, y_raw, meta = subset[int(dataset_idx)]
        # x is a PIL Image (torchvision transform not yet applied)
        yield x, int(label)


def get_source_hospital_dict(hospital_dict: dict):
    """
    Return a sub-dict containing only source (non-held-out) hospitals.

    Args
    ----
    hospital_dict : dict
        Full hospital dict from load_camelyon17().

    Returns
    -------
    dict : {hospital_id: ...} for source hospitals only.
    """
    return {
        h: v for h, v in hospital_dict.items()
        if h != config.HELD_OUT_HOSPITAL
    }


def get_ood_hospital_dict(hospital_dict: dict):
    """
    Return a sub-dict containing only the held-out OOD hospital.

    Args
    ----
    hospital_dict : dict
        Full hospital dict from load_camelyon17().

    Returns
    -------
    dict : {hospital_id: ...} for OOD hospital only.
    """
    h = config.HELD_OUT_HOSPITAL
    return {h: hospital_dict[h]}


def summarize_splits(hospital_dict: dict):
    """
    Print a formatted summary table of all hospital splits.

    Args
    ----
    hospital_dict : dict
        Output of load_camelyon17().
    """
    print("\n  Hospital split summary:")
    print(f"  {'Hospital':<25} {'N':>8} {'Tumor%':>8} {'Role':>12}")
    print("  " + "-" * 57)
    for h in sorted(hospital_dict.keys()):
        ys   = hospital_dict[h]['y']
        role = "OOD test" if h == config.HELD_OUT_HOSPITAL else "Source"
        print(f"  {config.HOSPITAL_NAMES[h]:<25} {len(ys):>8,} "
              f"{ys.mean()*100:>7.1f}% {role:>12}")
    print()
