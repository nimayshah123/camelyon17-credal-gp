"""
01_extract_features.py — Standalone: extract ResNet50 features and apply PCA.

Runs the full feature extraction pipeline:
  1. Downloads Camelyon17-WILDS (if not already present).
  2. Loads and splits dataset by hospital.
  3. Extracts 2048-dim ResNet50 embeddings per patch.
  4. Fits PCA on source hospitals (0,1,2,4) and transforms all 5.
  5. Saves .npy feature arrays to features/ directory.

Usage
-----
    python scripts/01_extract_features.py [--fast-dev] [--force]

Arguments
---------
--fast-dev : use 500 samples/hospital for fast testing.
--force    : re-extract even if feature files already exist.
"""

import os
import sys
import argparse
import time

# Add project root to path
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

import config
from src.data_loader        import load_camelyon17, summarize_splits
from src.feature_extractor  import (
    extract_and_save_all_features,
    features_exist_on_disk,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ResNet50 features from Camelyon17-WILDS."
    )
    parser.add_argument('--fast-dev', action='store_true',
                        help="Use 500 samples/hospital for fast testing.")
    parser.add_argument('--force', action='store_true',
                        help="Re-extract even if feature files exist.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides
    if args.fast_dev:
        config.FAST_DEV = True

    print(f"\n{'='*60}")
    print("STEP 1: FEATURE EXTRACTION")
    print(f"{'='*60}")
    print(f"  Dataset root : {config.DATASET_ROOT}")
    print(f"  Features dir : {config.FEATURES_DIR}")
    print(f"  PCA dims     : {config.PCA_DIMS}")
    print(f"  Batch size   : {config.BATCH_SIZE}")
    print(f"  Fast dev     : {config.FAST_DEV}")
    if config.FAST_DEV:
        print(f"  Samples/hosp : {config.FAST_DEV_N}")
    print(f"  Force re-extract: {args.force}")

    # Check if we can skip
    if features_exist_on_disk() and not args.force:
        print("\n  Feature files already exist. Skipping extraction.")
        print("  (Use --force to re-extract.)")
        return

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    t0 = time.time()
    print("\n  Loading Camelyon17-WILDS …")
    hospital_dict, dataset_obj = load_camelyon17(
        root=config.DATASET_ROOT,
        fast_dev=config.FAST_DEV,
    )
    summarize_splits(hospital_dict)
    print(f"  Dataset loaded in {time.time()-t0:.1f}s")

    # -----------------------------------------------------------------------
    # Extract features
    # -----------------------------------------------------------------------
    t1 = time.time()
    print("\n  Extracting features (ResNet50 → PCA) …")
    features_dict, labels_dict = extract_and_save_all_features(
        hospital_dict, force=args.force
    )
    elapsed = time.time() - t1

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print(f"\n  Feature extraction complete in {elapsed:.1f}s")
    print(f"\n  Feature shapes:")
    for h in range(config.N_HOSPITALS):
        print(f"    Hospital {h}: {features_dict[h].shape}  "
              f"labels: {labels_dict[h].shape}")

    print(f"\n  Files saved to: {config.FEATURES_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
