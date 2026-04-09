"""
feature_extractor.py — ResNet50 deep feature extraction and PCA reduction.

Pipeline
--------
1. Load pretrained ResNet50 (ImageNet weights), strip the final FC layer.
2. Pass each hospital's patches through avgpool → 2048-dim embedding.
3. Fit PCA on source hospitals (0,1,2,4), transform all 5 hospitals to 64-d.
4. Save per-hospital features to disk as .npy arrays.
5. On subsequent runs, load from disk if files already exist.

The PCA fitting is intentionally done on SOURCE hospitals only to simulate a
realistic OOD scenario where the target domain (hospital 3) is unseen during
the feature-reduction step.
"""

import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ---------------------------------------------------------------------------
# Torch imports (deferred so the module can be imported without torch installed
# as long as load_features() is only called)
# ---------------------------------------------------------------------------
_TORCH_AVAILABLE = False
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
    from sklearn.decomposition import PCA
    import joblib
    _TORCH_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Image transform matching ResNet50 ImageNet preprocessing
# ---------------------------------------------------------------------------
_IMAGENET_TRANSFORM = None

def _get_transform():
    global _IMAGENET_TRANSFORM
    if _IMAGENET_TRANSFORM is None:
        _IMAGENET_TRANSFORM = T.Compose([
            T.Resize(112),
            T.CenterCrop(96),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    return _IMAGENET_TRANSFORM


# ---------------------------------------------------------------------------
# Custom Dataset wrapper
# ---------------------------------------------------------------------------
class HospitalPatchDataset(Dataset):
    """
    PyTorch Dataset wrapping a hospital's patches for DataLoader batching.

    Args
    ----
    subset : wilds Subset object
        The WILDS subset (e.g. train_data).
    indices : np.ndarray
        Integer indices into subset to include.
    labels : np.ndarray
        Corresponding labels.
    transform : callable, optional
        Image transform.
    """

    def __init__(self, subset, indices, labels, transform=None):
        self.subset    = subset
        self.indices   = indices
        self.labels    = labels
        self.transform = transform or _get_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos):
        dataset_idx   = int(self.indices[pos])
        label         = int(self.labels[pos])
        x, _, _       = self.subset[dataset_idx]

        # x may already be a Tensor if WILDS applies its own transform;
        # convert back to PIL so our transform can work uniformly.
        if hasattr(x, 'numpy'):
            from PIL import Image
            x_np = x.numpy()
            if x_np.dtype != np.uint8:
                x_np = (x_np * 255).clip(0, 255).astype(np.uint8)
            if x_np.ndim == 3 and x_np.shape[0] == 3:
                x_np = x_np.transpose(1, 2, 0)
            x = Image.fromarray(x_np)

        x_tensor = self.transform(x)
        return x_tensor, label


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------
def build_resnet50_extractor():
    """
    Build a frozen ResNet50 feature extractor (removes final FC layer).

    Returns
    -------
    model : torch.nn.Module
        ResNet50 up to avgpool, outputting 2048-dim vectors.
    device : torch.device
        Best available device (cuda if available, else cpu).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch / torchvision not installed.")

    model  = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove final classification layer — keep everything up to avgpool
    model  = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    print(f"  ResNet50 extractor built on device: {device}")
    return model, device


def extract_embeddings_for_hospital(
    model,
    device,
    hospital_data: dict,
    hospital_id: int,
    batch_size: int = None,
) -> np.ndarray:
    """
    Extract 2048-dim embeddings for all patches in one hospital.

    Args
    ----
    model : torch.nn.Module
        Frozen ResNet50 extractor.
    device : torch.device
    hospital_data : dict
        Entry from load_camelyon17() output.
    hospital_id : int
    batch_size : int, optional
        Defaults to config.BATCH_SIZE.

    Returns
    -------
    embeddings : np.ndarray, shape (N, 2048)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch not installed.")

    batch_size = batch_size or config.BATCH_SIZE
    h_dict     = hospital_data[hospital_id]
    subset     = h_dict['dataset_ref']
    indices    = h_dict['subset_indices']
    labels     = h_dict['y']

    ds     = HospitalPatchDataset(subset, indices, labels)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,    # 0 workers for Windows compatibility
        pin_memory=(device.type == 'cuda'),
    )

    all_embeddings = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader,
                            desc=f"    Hospital {hospital_id}",
                            leave=False):
            imgs = imgs.to(device)
            out  = model(imgs)                  # (B, 2048, 1, 1)
            out  = out.squeeze(-1).squeeze(-1)  # (B, 2048)
            all_embeddings.append(out.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def extract_and_save_all_features(hospital_dict: dict, force: bool = False):
    """
    Extract ResNet50 embeddings for all 5 hospitals, apply PCA, save to disk.

    Skips extraction if .npy files already exist (unless force=True).

    Args
    ----
    hospital_dict : dict
        Output of data_loader.load_camelyon17().
    force : bool
        Re-extract even if files already exist.

    Returns
    -------
    features_dict : dict
        {hospital_id: np.ndarray shape (N, PCA_DIMS)}
    labels_dict : dict
        {hospital_id: np.ndarray shape (N,)}
    """
    os.makedirs(config.FEATURES_DIR, exist_ok=True)

    # Check if all PCA feature files exist already
    all_exist = all(
        os.path.exists(
            os.path.join(config.FEATURES_DIR, f"hospital_{h}_features.npy")
        )
        for h in range(config.N_HOSPITALS)
    )

    labels_dict = {h: hospital_dict[h]['y'] for h in range(config.N_HOSPITALS)}

    if all_exist and not force:
        print("  Feature files found on disk — loading without re-extraction.")
        return load_features_from_disk(), labels_dict

    print("  Building ResNet50 extractor …")
    model, device = build_resnet50_extractor()

    # -----------------------------------------------------------------------
    # Step 1: Extract raw 2048-d embeddings for all hospitals
    # -----------------------------------------------------------------------
    raw_embeddings = {}
    for h in range(config.N_HOSPITALS):
        emb_path = os.path.join(config.FEATURES_DIR,
                                f"hospital_{h}_raw_embeddings.npy")
        if os.path.exists(emb_path) and not force:
            print(f"  Hospital {h}: raw embeddings found, loading …")
            raw_embeddings[h] = np.load(emb_path)
        else:
            print(f"  Hospital {h}: extracting embeddings …")
            emb = extract_embeddings_for_hospital(model, device,
                                                  hospital_dict, h)
            np.save(emb_path, emb)
            raw_embeddings[h] = emb
            print(f"    → shape {emb.shape}")

    # -----------------------------------------------------------------------
    # Step 2: Fit PCA on source hospitals only, transform all 5
    # -----------------------------------------------------------------------
    print(f"\n  Fitting PCA ({config.PCA_DIMS}d) on source hospitals "
          f"{config.SOURCE_HOSPITALS} …")

    src_embeddings = np.concatenate(
        [raw_embeddings[h] for h in config.SOURCE_HOSPITALS], axis=0
    )

    rng = np.random.RandomState(config.RANDOM_SEED)
    pca = PCA(n_components=config.PCA_DIMS, random_state=config.RANDOM_SEED)
    pca.fit(src_embeddings)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA explained variance ({config.PCA_DIMS} components): "
          f"{explained:.1f}%")

    # Save PCA model
    pca_path = os.path.join(config.FEATURES_DIR, "pca_model.pkl")
    joblib.dump(pca, pca_path)
    print(f"  PCA model saved to {pca_path}")

    # -----------------------------------------------------------------------
    # Step 3: Transform all hospitals and save
    # -----------------------------------------------------------------------
    features_dict = {}
    for h in range(config.N_HOSPITALS):
        feats = pca.transform(raw_embeddings[h])
        path  = os.path.join(config.FEATURES_DIR,
                             f"hospital_{h}_features.npy")
        np.save(path, feats)
        features_dict[h] = feats
        print(f"  Hospital {h}: PCA features saved → {path} "
              f"(shape {feats.shape})")

    # Save labels too for convenience
    for h in range(config.N_HOSPITALS):
        lbl_path = os.path.join(config.FEATURES_DIR,
                                f"hospital_{h}_labels.npy")
        np.save(lbl_path, labels_dict[h])

    return features_dict, labels_dict


def load_features_from_disk():
    """
    Load PCA-reduced feature arrays from disk.

    Returns
    -------
    features_dict : dict
        {hospital_id: np.ndarray shape (N, PCA_DIMS)}

    Raises
    ------
    FileNotFoundError if any expected file is missing.
    """
    features_dict = {}
    for h in range(config.N_HOSPITALS):
        path = os.path.join(config.FEATURES_DIR,
                            f"hospital_{h}_features.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Feature file not found: {path}\n"
                "Run scripts/01_extract_features.py first."
            )
        features_dict[h] = np.load(path)
    return features_dict


def load_labels_from_disk():
    """
    Load label arrays from disk.

    Returns
    -------
    labels_dict : dict
        {hospital_id: np.ndarray shape (N,)}
    """
    labels_dict = {}
    for h in range(config.N_HOSPITALS):
        path = os.path.join(config.FEATURES_DIR,
                            f"hospital_{h}_labels.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Labels file not found: {path}\n"
                "Run scripts/01_extract_features.py first."
            )
        labels_dict[h] = np.load(path)
    return labels_dict


def features_exist_on_disk() -> bool:
    """Return True if all PCA feature .npy files exist."""
    return all(
        os.path.exists(
            os.path.join(config.FEATURES_DIR, f"hospital_{h}_features.npy")
        )
        for h in range(config.N_HOSPITALS)
    )
