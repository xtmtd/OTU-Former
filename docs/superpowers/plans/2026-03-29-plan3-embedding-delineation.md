# OTU-Former Plan 3: Embedding + Delineation Module

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `embedding/` and `delineation/` subpackages and wire up `extract`, `evaluate`, `cluster`, `annotate`, `diversity` CLI commands. This delivers the core analysis pipeline: checkpoint → embeddings → UPGMA tree → OTU partitions → diversity indices.

**Architecture:** `embedding/extractor.py` handles checkpoint → embeddings CSV (single + batch mode). `embedding/evaluator.py` handles all embedding quality metrics + UMAP. `delineation/` handles distance computation, UPGMA tree, partition scanning, expert annotation write-back, and alpha diversity. All logic migrated from `ref/embeddings_tree20260206.py` and the diversity shell script, ported to pure Python.

**Tech Stack:** PyTorch, timm, numpy, scipy, scikit-learn, scikit-bio, umap-learn, matplotlib, seaborn, pandas

**Prerequisites:** Plan 1 (CLI stubs), Plan 2 (training model — needed for checkpoint loading).

**Reference sources:**
- `ref/ibot20260115.py` — extract mode logic (lines ~3001–3038 args, plus extract functions)
- `ref/embeddings_tree20260206.py` — distance, tree, partition logic
- `ref/diversity_index.txt` — diversity calculation workflow

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/otuformer/embedding/extractor.py` | Load checkpoint → run inference → write embeddings CSV; single + batch mode |
| `src/otuformer/embedding/evaluator.py` | kNN, linear probe, Recall@K, mAP, NMI/ARI/AMI, Silhouette, Purity, UMAP |
| `src/otuformer/delineation/distance.py` | Cosine + Euclidean pairwise distance; PCA whitening; k-NN local scaling; chunked computation |
| `src/otuformer/delineation/tree.py` | UPGMA linkage matrix; bootstrap support; Newick export |
| `src/otuformer/delineation/partition.py` | Two-stage dynamic threshold scan; partition export (assignments + summary CSV); partition metrics (NMI, ARI, BCubed-F, V-measure, Silhouette) per cutoff |
| `src/otuformer/delineation/annotate.py` | Expert correction write-back; annotation summary |
| `src/otuformer/delineation/diversity.py` | Alpha diversity indices (all categories); MPD via scikit-bio |
| `src/otuformer/cli/extract.py` | Replace stub: call `run_extract()` |
| `src/otuformer/cli/evaluate.py` | Replace stub: call `run_evaluate()` |
| `src/otuformer/cli/cluster.py` | Replace stub: call `run_cluster()` |
| `src/otuformer/cli/annotate.py` | Replace stub: call `run_annotate()` |
| `src/otuformer/cli/diversity.py` | Replace stub: call `run_diversity()` |
| `tests/test_extractor.py` | Unit tests for embedding extraction |
| `tests/test_evaluator.py` | Unit tests for embedding evaluation metrics |
| `tests/test_distance.py` | Unit tests for distance computation |
| `tests/test_tree.py` | Unit tests for UPGMA construction |
| `tests/test_partition.py` | Unit tests for partition scanning |
| `tests/test_annotate.py` | Unit tests for annotation write-back |
| `tests/test_diversity.py` | Unit tests for diversity indices |

---

## Task 1: `embedding/extractor.py`

**Files:**
- Create: `src/otuformer/embedding/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_extractor.py
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from otuformer.embedding.extractor import extract_embeddings, detect_batch_mode


def make_checkpoint(tmp_path, out_dim=64):
    """Save a minimal checkpoint."""
    from otuformer.training.model import OTUFormerEncoder
    model = OTUFormerEncoder(model_name="vit_tiny_patch16_224", out_dim=out_dim)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {"model_name": "vit_tiny_patch16_224", "out_dim": out_dim},
    }
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, p)
    return p


def make_images(directory, n=3):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (32, 32), color=(i * 80, 0, 0)).save(directory / f"img_{i}.jpg")


def test_extract_single_dir(tmp_path):
    ckpt = make_checkpoint(tmp_path)
    img_dir = tmp_path / "images"
    make_images(img_dir)
    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=img_dir,
        model_name="vit_tiny_patch16_224",
        extract_size=32,
        batch_size=2,
        device="cpu",
    )
    assert isinstance(out, pd.DataFrame)
    assert "id" in out.columns
    assert len(out) == 3
    assert out.shape[1] > 2  # id + embedding dims


def test_detect_batch_mode(tmp_path):
    # Single directory with images → not batch
    img_dir = tmp_path / "single"
    make_images(img_dir)
    assert detect_batch_mode(img_dir) is False

    # Parent directory with subdirectories → batch
    parent = tmp_path / "multi"
    make_images(parent / "site_a")
    make_images(parent / "site_b")
    assert detect_batch_mode(parent) is True


def test_extract_batch_mode(tmp_path):
    ckpt = make_checkpoint(tmp_path)
    parent = tmp_path / "multi"
    make_images(parent / "site_a", n=2)
    make_images(parent / "site_b", n=3)
    out = extract_embeddings(
        checkpoint_path=ckpt,
        images_dir=parent,
        model_name="vit_tiny_patch16_224",
        extract_size=32,
        batch_size=2,
        device="cpu",
    )
    assert "sample" in out.columns
    assert len(out) == 5
    assert set(out["sample"].unique()) == {"site_a", "site_b"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_extractor.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `embedding/extractor.py`**

```python
# src/otuformer/embedding/extractor.py
"""Extract embeddings from images using a trained OTU-Former checkpoint."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from otuformer.training.model import OTUFormerEncoder
from otuformer.utils.checkpoint import load_checkpoint

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class ImageFolderDataset(Dataset):
    """Load all images from a directory."""

    def __init__(self, images_dir: Path, extract_size: int = 224) -> None:
        self.paths = sorted(
            p for p in images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.transform = transforms.Compose([
            transforms.Resize(int(extract_size * 1.14)),
            transforms.CenterCrop(extract_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx].name


def detect_batch_mode(images_dir: Path) -> bool:
    """Return True if images_dir contains subdirectories with images (batch mode)."""
    subdirs = [p for p in images_dir.iterdir() if p.is_dir()]
    for sub in subdirs:
        if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in sub.iterdir() if p.is_file()):
            return True
    return False


def _load_model(checkpoint_path: Path, model_name: str, device: torch.device) -> OTUFormerEncoder:
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt.get("config", {})
    out_dim = cfg.get("out_dim") or cfg.get("metric_embed_dim", 256)
    resolved_model = cfg.get("model_name", model_name)
    model = OTUFormerEncoder(model_name=resolved_model, out_dim=out_dim)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)
    return model


def _extract_one_dir(
    model: torch.nn.Module,
    images_dir: Path,
    extract_size: int,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> pd.DataFrame:
    ds = ImageFolderDataset(images_dir, extract_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_ids, all_embs = [], []
    with torch.no_grad():
        for imgs, names in tqdm(loader, desc=f"Extracting {images_dir.name}"):
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            all_ids.extend(names)
            all_embs.append(embs)
    embs_array = np.concatenate(all_embs, axis=0)
    dim_cols = [f"dim_{i}" for i in range(embs_array.shape[1])]
    df = pd.DataFrame(embs_array, columns=dim_cols)
    df.insert(0, "id", all_ids)
    return df


def extract_embeddings(
    checkpoint_path: Path,
    images_dir: Path,
    model_name: str = "vit_small_patch16_224",
    extract_size: int = 224,
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
) -> pd.DataFrame:
    """Extract embeddings. Returns DataFrame with 'id' column + embedding dims.

    In batch mode (subdirectories detected), adds 'sample' column.
    """
    dev = torch.device(device)
    model = _load_model(checkpoint_path, model_name, dev)

    if detect_batch_mode(images_dir):
        subdirs = sorted(p for p in images_dir.iterdir() if p.is_dir())
        frames = []
        for sub in subdirs:
            df = _extract_one_dir(model, sub, extract_size, batch_size, dev, num_workers)
            df.insert(1, "sample", sub.name)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
    else:
        return _extract_one_dir(model, images_dir, extract_size, batch_size, dev, num_workers)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_extractor.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/embedding/extractor.py tests/test_extractor.py
git commit -m "feat: add embedding extractor with batch mode support"
```

---

## Task 2: `embedding/evaluator.py`

**Files:**
- Create: `src/otuformer/embedding/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluator.py
import numpy as np
import pytest
from otuformer.embedding.evaluator import (
    compute_knn_accuracy,
    compute_recall_at_k,
    compute_map,
    compute_clustering_metrics,
    compute_metric_learning_diagnostics,
)


def make_data(n=50, dim=32, n_classes=5):
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(n_classes), n // n_classes)
    embeddings = rng.standard_normal((n, dim))
    # Make within-class embeddings similar
    for c in range(n_classes):
        mask = labels == c
        embeddings[mask] += rng.standard_normal(dim) * 3
    # L2 normalize
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings, labels


def test_knn_accuracy_returns_dict():
    embs, labels = make_data()
    result = compute_knn_accuracy(embs, labels, k_values=[1, 5])
    assert "kNN_Acc_k1" in result
    assert "kNN_Acc_k5" in result
    assert 0.0 <= result["kNN_Acc_k1"] <= 1.0


def test_recall_at_k_returns_dict():
    embs, labels = make_data()
    result = compute_recall_at_k(embs, labels, k_values=[1, 5, 10])
    assert "Recall@1" in result
    assert 0.0 <= result["Recall@1"] <= 1.0


def test_clustering_metrics_keys():
    embs, labels = make_data()
    result = compute_clustering_metrics(embs, labels)
    for key in ["NMI", "ARI", "AMI", "Silhouette", "Purity"]:
        assert key in result


def test_metric_learning_diagnostics_keys():
    embs, labels = make_data()
    result = compute_metric_learning_diagnostics(embs, labels)
    for key in ["Intra_Class_Var", "Inter_Class_Dist", "Embedding_Norm_Mean", "Embedding_Norm_Std"]:
        assert key in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_evaluator.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `embedding/evaluator.py`**

Port metric computation functions from `ref/ibot20260115.py` (lines ~468–700). Add UMAP visualisation:

```python
# src/otuformer/embedding/evaluator.py
"""Embedding quality evaluation: kNN, Recall@K, mAP, clustering, UMAP."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional
import pandas as pd


def compute_knn_accuracy(
    embeddings: np.ndarray, labels: np.ndarray, k_values: list[int] = [1, 5, 10]
) -> dict[str, float]:
    """k-NN classification accuracy via leave-one-out."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import normalize
    X = normalize(embeddings, norm="l2")
    result = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        scores = cross_val_score(knn, X, labels, cv=min(5, len(np.unique(labels))))
        result[f"kNN_Acc_k{k}"] = float(scores.mean())
    return result


def compute_recall_at_k(
    embeddings: np.ndarray, labels: np.ndarray, k_values: list[int] = [1, 5, 10]
) -> dict[str, float]:
    """Recall@K for retrieval (exclude self)."""
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    X = normalize(embeddings, norm="l2")
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -np.inf)
    result = {}
    for k in k_values:
        k_actual = min(k, len(labels) - 1)
        correct = sum(
            labels[i] in labels[np.argsort(sim[i])[-k_actual:]]
            for i in range(len(labels))
        )
        result[f"Recall@{k}"] = correct / len(labels)
    return result


def compute_map(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean Average Precision for retrieval."""
    from sklearn.preprocessing import normalize
    from sklearn.metrics.pairwise import cosine_similarity
    X = normalize(embeddings, norm="l2")
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -np.inf)
    aps = []
    for i in range(len(labels)):
        ranked = np.argsort(sim[i])[::-1]
        relevant = labels[ranked] == labels[i]
        n_rel = relevant.sum()
        if n_rel == 0:
            continue
        precisions = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        aps.append((precisions * relevant).sum() / n_rel)
    return float(np.mean(aps)) if aps else 0.0


def compute_clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """NMI, ARI, AMI, Silhouette, Purity via k-Means with true k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        adjusted_mutual_info_score,
        silhouette_score,
    )
    n_clusters = len(np.unique(labels))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = km.fit_predict(embeddings)
    purity = sum(
        np.bincount(labels[pred == c]).max()
        for c in range(n_clusters)
    ) / len(labels)
    return {
        "NMI": float(normalized_mutual_info_score(labels, pred)),
        "ARI": float(adjusted_rand_score(labels, pred)),
        "AMI": float(adjusted_mutual_info_score(labels, pred)),
        "Silhouette": float(silhouette_score(embeddings, pred)),
        "Purity": float(purity),
    }


def compute_metric_learning_diagnostics(
    embeddings: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Intra/inter-class distances and embedding norms."""
    norms = np.linalg.norm(embeddings, axis=1)
    unique_labels = np.unique(labels)
    intra_vars, inter_dists = [], []
    from sklearn.preprocessing import normalize
    X = normalize(embeddings, norm="l2")
    for c in unique_labels:
        mask = labels == c
        if mask.sum() > 1:
            class_embs = X[mask]
            sim = class_embs @ class_embs.T
            intra_vars.append(float(1 - sim[np.triu_indices(len(class_embs), k=1)].mean()))
    for i, c1 in enumerate(unique_labels):
        for c2 in unique_labels[i + 1:]:
            e1 = X[labels == c1].mean(axis=0)
            e2 = X[labels == c2].mean(axis=0)
            inter_dists.append(float(1 - e1 @ e2))
    return {
        "Intra_Class_Var": float(np.mean(intra_vars)) if intra_vars else 0.0,
        "Inter_Class_Dist": float(np.mean(inter_dists)) if inter_dists else 0.0,
        "Embedding_Norm_Mean": float(norms.mean()),
        "Embedding_Norm_Std": float(norms.std()),
    }


def compute_linear_probing(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Linear probing accuracy via cross-validated logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import normalize
    X = normalize(embeddings, norm="l2")
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, X, labels, cv=min(5, len(np.unique(labels))))
    return float(scores.mean())


def run_umap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray],
    out_path: Path,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    max_classes: int = 20,
) -> None:
    """Compute UMAP and save scatter plot PDF."""
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    if labels is not None:
        unique_labels = np.unique(labels)
        if max_classes > 0 and len(unique_labels) > max_classes:
            unique_labels = unique_labels[:max_classes]
        cmap = plt.cm.get_cmap("tab20", len(unique_labels))
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(proj[mask, 0], proj[mask, 1], s=5, color=cmap(i), label=str(lbl), alpha=0.7)
        if len(unique_labels) <= 20:
            ax.legend(markerscale=3, fontsize=7, ncol=2)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.5)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2" if n_components >= 2 else "")
    ax.set_title("UMAP Embedding Visualisation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_evaluator.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/embedding/evaluator.py tests/test_evaluator.py
git commit -m "feat: add embedding evaluator (kNN, Recall@K, mAP, clustering, UMAP)"
```

---

## Task 3: `delineation/distance.py`

Port from `ref/embeddings_tree20260206.py` (`compute_cosine_distances`, `apply_local_scaling`, `apply_pca_whitening`). Add Euclidean distance.

**Files:**
- Create: `src/otuformer/delineation/distance.py`
- Create: `tests/test_distance.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_distance.py
import numpy as np
import pytest
from otuformer.delineation.distance import (
    compute_cosine_distances,
    compute_euclidean_distances,
    apply_local_scaling,
    apply_pca_whitening,
)


def random_embeddings(n=20, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float64)


def test_cosine_distances_shape():
    X = random_embeddings()
    D = compute_cosine_distances(X)
    assert D.shape == (20, 20)


def test_cosine_distances_symmetric():
    X = random_embeddings()
    D = compute_cosine_distances(X)
    assert np.allclose(D, D.T, atol=1e-6)


def test_cosine_distances_zero_diagonal():
    X = random_embeddings()
    D = compute_cosine_distances(X)
    assert np.allclose(np.diag(D), 0.0)


def test_cosine_distances_range():
    X = random_embeddings()
    D = compute_cosine_distances(X)
    assert D.min() >= -1e-6
    assert D.max() <= 2.0 + 1e-6


def test_euclidean_distances_shape():
    X = random_embeddings()
    D = compute_euclidean_distances(X)
    assert D.shape == (20, 20)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T, atol=1e-6)


def test_local_scaling_changes_distances():
    X = random_embeddings(n=30)
    D = compute_cosine_distances(X)
    D_scaled, _ = apply_local_scaling(D, k=5)
    assert not np.allclose(D, D_scaled)
    assert np.allclose(np.diag(D_scaled), 0.0)


def test_pca_whitening_reduces_dims():
    X = random_embeddings(n=50, dim=64)
    X_pca, _ = apply_pca_whitening(X, n_components=16)
    assert X_pca.shape == (50, 16)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_distance.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `delineation/distance.py`**

Port and extend from `ref/embeddings_tree20260206.py`:

```python
# src/otuformer/delineation/distance.py
"""Pairwise distance computation: cosine, Euclidean, PCA whitening, local scaling."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def _safe_l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def compute_cosine_distances(
    X: np.ndarray, chunk_size: int = 5000
) -> np.ndarray:
    """Pairwise cosine distances. Returns symmetric float64 matrix in [0, 2]."""
    n = X.shape[0]
    if n < 10000:
        X_norm = _safe_l2_normalize(X.astype(np.float64))
        D = 1.0 - X_norm @ X_norm.T
    else:
        X_norm = _safe_l2_normalize(X.astype(np.float32))
        D = np.zeros((n, n), dtype=np.float32)
        n_chunks = (n + chunk_size - 1) // chunk_size
        for i in range(n_chunks):
            si, ei = i * chunk_size, min((i + 1) * chunk_size, n)
            for j in range(i, n_chunks):
                sj, ej = j * chunk_size, min((j + 1) * chunk_size, n)
                block = 1.0 - X_norm[si:ei] @ X_norm[sj:ej].T
                D[si:ei, sj:ej] = block
                if i != j:
                    D[sj:ej, si:ei] = block.T
        D = D.astype(np.float64)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    return np.clip(D, 0.0, 2.0)


def compute_euclidean_distances(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distances. Returns symmetric float64 matrix."""
    from scipy.spatial.distance import cdist
    D = cdist(X.astype(np.float64), X.astype(np.float64), metric="euclidean")
    np.fill_diagonal(D, 0.0)
    return (D + D.T) / 2.0


def apply_pca_whitening(
    X: np.ndarray, n_components: int, random_state: int = 42
) -> tuple[np.ndarray, PCA]:
    """Fit PCA whitening on X and return transformed data + fitted PCA."""
    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def apply_local_scaling(
    dist_matrix: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """k-NN local scaling: d_scaled(i,j) = d(i,j) / sqrt(sigma_i * sigma_j).

    sigma_i = distance to k-th nearest neighbour of point i.
    Ported from ref/embeddings_tree20260206.py.
    """
    knn_dists = np.partition(dist_matrix, k, axis=1)[:, k]
    scale = np.sqrt(np.outer(knn_dists, knn_dists))
    scale = np.maximum(scale, 1e-10)
    scaled = dist_matrix / scale
    np.fill_diagonal(scaled, 0.0)
    scaled = (scaled + scaled.T) / 2.0
    return scaled, knn_dists


def auto_select_k(n_samples: int, strategy: str = "adaptive") -> int:
    """Auto-select k for local scaling based on dataset size."""
    import math
    if strategy == "fixed":
        k = 7
    elif strategy == "sqrt":
        k = int(math.sqrt(n_samples))
    elif strategy == "log":
        k = max(5, int(5 * math.log2(n_samples)))
    else:  # adaptive
        if n_samples < 100:
            k = 7
        elif n_samples < 500:
            k = int(math.sqrt(n_samples))
        else:
            k = int(5 * math.log2(n_samples))
        k = max(5, min(k, 100))
    return min(k, n_samples - 1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_distance.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/distance.py tests/test_distance.py
git commit -m "feat: add cosine/euclidean distance, PCA whitening, local scaling"
```

---

## Task 4: `delineation/tree.py` — UPGMA + bootstrap

**Files:**
- Create: `src/otuformer/delineation/tree.py`
- Create: `tests/test_tree.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tree.py
import numpy as np
from otuformer.delineation.tree import build_upgma, upgma_to_newick


def simple_dist_matrix():
    # 4 samples with known structure
    D = np.array([
        [0.0, 0.1, 0.8, 0.9],
        [0.1, 0.0, 0.7, 0.8],
        [0.8, 0.7, 0.0, 0.1],
        [0.9, 0.8, 0.1, 0.0],
    ])
    return D


def test_build_upgma_returns_linkage():
    D = simple_dist_matrix()
    Z = build_upgma(D)
    assert Z.shape == (3, 4)  # n-1 merges, 4 columns


def test_upgma_newick_output(tmp_path):
    D = simple_dist_matrix()
    ids = ["A", "B", "C", "D"]
    Z = build_upgma(D)
    nwk_path = tmp_path / "tree.nwk"
    upgma_to_newick(Z, ids, nwk_path)
    content = nwk_path.read_text()
    assert content.endswith(";")
    for id_ in ids:
        assert id_ in content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_tree.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `delineation/tree.py`**

```python
# src/otuformer/delineation/tree.py
"""UPGMA tree construction and Newick export."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


def build_upgma(dist_matrix: np.ndarray) -> np.ndarray:
    """Build UPGMA linkage matrix from square distance matrix."""
    condensed = squareform(dist_matrix, checks=False)
    return hierarchy.linkage(condensed, method="average")


def upgma_to_newick(
    Z: np.ndarray,
    ids: list[str],
    out_path: Path,
) -> str:
    """Convert scipy linkage matrix to Newick string and write to file."""
    n = len(ids)
    # Build nested structure bottom-up
    nodes: dict[int, str] = {i: ids[i] for i in range(n)}
    heights: dict[int, float] = {i: 0.0 for i in range(n)}

    for step, (a, b, dist, _) in enumerate(Z):
        a, b = int(a), int(b)
        node_id = n + step
        height = dist / 2.0
        branch_a = height - heights[a]
        branch_b = height - heights[b]
        nodes[node_id] = f"({nodes[a]}:{branch_a:.6f},{nodes[b]}:{branch_b:.6f})"
        heights[node_id] = height

    newick = nodes[n + len(Z) - 1] + ";"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(newick)
    return newick


def compute_bootstrap_support(
    dist_matrix: np.ndarray,
    ids: list[str],
    n_bootstraps: int = 100,
    subsample_ratio: float = 0.8,
    random_state: int = 42,
) -> dict[frozenset, float]:
    """Compute bootstrap support for UPGMA clades.

    Returns dict mapping frozenset of tip names → support percentage.
    """
    from otuformer.delineation.distance import compute_cosine_distances

    rng = np.random.default_rng(random_state)
    n_dims = dist_matrix.shape[0]

    # Build reference clades from original tree
    Z_ref = build_upgma(dist_matrix)
    ref_clades = _extract_clades(Z_ref, ids)

    clade_counts: dict[frozenset, int] = {c: 0 for c in ref_clades}

    for _ in range(n_bootstraps):
        n_cols = int(n_dims * subsample_ratio)
        col_idx = rng.choice(dist_matrix.shape[1], size=n_cols, replace=True)
        # Resample columns of the embedding space (approximate)
        D_boot = dist_matrix[:, col_idx] if dist_matrix.ndim == 2 else dist_matrix
        Z_boot = build_upgma(D_boot)  # NOTE: must use D_boot, not original dist_matrix
        boot_clades = _extract_clades(Z_boot, ids)
        for clade in ref_clades:
            if clade in boot_clades:
                clade_counts[clade] += 1

    return {c: 100.0 * cnt / n_bootstraps for c, cnt in clade_counts.items()}


def _extract_clades(Z: np.ndarray, ids: list[str]) -> set[frozenset]:
    """Extract all non-trivial clades from linkage matrix."""
    n = len(ids)
    clusters: dict[int, frozenset] = {i: frozenset([ids[i]]) for i in range(n)}
    clades: set[frozenset] = set()
    for step, (a, b, _, _) in enumerate(Z):
        a, b = int(a), int(b)
        merged = clusters[a] | clusters[b]
        if 1 < len(merged) < n:
            clades.add(merged)
        clusters[n + step] = merged
    return clades
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_tree.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/tree.py tests/test_tree.py
git commit -m "feat: add UPGMA tree construction and Newick export"
```

---

## Task 5: `delineation/partition.py` — two-stage threshold scan

Port from `ref/embeddings_tree20260206.py`.

**Files:**
- Create: `src/otuformer/delineation/partition.py`
- Create: `tests/test_partition.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_partition.py
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from otuformer.delineation.partition import (
    build_partitions_from_linkage,
    export_partition_tables,
    compute_partition_metrics,
    two_stage_threshold_scan,
)


def simple_linkage():
    D = np.array([
        [0.0, 0.1, 0.8, 0.9],
        [0.1, 0.0, 0.7, 0.8],
        [0.8, 0.7, 0.0, 0.1],
        [0.9, 0.8, 0.1, 0.0],
    ])
    return hierarchy.linkage(squareform(D, checks=False), method="average"), ["A","B","C","D"]


def test_partitions_returned():
    Z, ids = simple_linkage()
    parts = build_partitions_from_linkage(Z, [0.3, 0.5])
    assert len(parts) == 2
    for th, labels in parts.items():
        assert len(labels) == 4


def test_export_creates_files(tmp_path):
    Z, ids = simple_linkage()
    parts = build_partitions_from_linkage(Z, [0.5])
    export_partition_tables(ids, parts, tmp_path, prefix="OTU")
    files = list(tmp_path.glob("*.csv"))
    assert len(files) >= 2


def test_partition_prefix_applied(tmp_path):
    Z, ids = simple_linkage()
    parts = build_partitions_from_linkage(Z, [0.5])
    export_partition_tables(ids, parts, tmp_path, prefix="Dun2024")
    summary = pd.read_csv(next(tmp_path.glob("*summary*")))
    assert all(summary["cluster"].str.startswith("Dun2024"))


def test_partition_metrics_keys():
    Z, ids = simple_linkage()
    labels_true = np.array([0, 0, 1, 1])
    parts = build_partitions_from_linkage(Z, [0.5])
    metrics = compute_partition_metrics(parts, labels_true)
    assert 0.5 in metrics
    row = metrics[0.5]
    for k in ["NMI", "ARI"]:
        assert k in row
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_partition.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `delineation/partition.py`**

Port from `ref/embeddings_tree20260206.py`. Key functions: `build_partitions_from_linkage`, `scan_linkage_thresholds`, `export_partition_tables`, `compute_partition_metrics`, `two_stage_threshold_scan`, and the tree+partition visualisation (`plot_upgma_partition_tree_panel`).

```python
# src/otuformer/delineation/partition.py
"""Partition scanning, export, and quality metrics for UPGMA tree."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy


def build_partitions_from_linkage(
    Z: np.ndarray, thresholds: list[float]
) -> OrderedDict[float, np.ndarray]:
    """Build cluster label arrays for each distance threshold."""
    parts: OrderedDict[float, np.ndarray] = OrderedDict()
    for th in sorted(thresholds):
        parts[float(th)] = hierarchy.fcluster(Z, t=th, criterion="distance")
    return parts


def export_partition_tables(
    ids: list[str],
    partitions: OrderedDict[float, np.ndarray],
    out_dir: Path,
    prefix: str = "OTU",
) -> None:
    """Write per-cutoff assignments + summary CSVs with OTU prefix naming."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for th, labels in partitions.items():
        clusters: dict[int, list[str]] = defaultdict(list)
        for lab, tip in zip(labels, ids):
            clusters[lab].append(tip)

        sorted_clusters = sorted(clusters.items(), key=lambda x: (-len(x[1]), x[1][0]))

        rows_summary, rows_assign = [], []
        for idx, (_, tips) in enumerate(sorted_clusters, start=1):
            cname = f"{prefix}_{idx}" if prefix else f"OTU_{idx}"
            rows_summary.append({"cluster": cname, "size": len(tips), "members": ";".join(tips)})
            for tip in tips:
                rows_assign.append({"id": tip, "cluster": cname})

        th_tag = f"{th:.4f}".rstrip("0").rstrip(".")
        pd.DataFrame(rows_summary).to_csv(out_dir / f"partition_{th_tag}_summary.csv", index=False)
        pd.DataFrame(rows_assign).to_csv(out_dir / f"partition_{th_tag}_assignments.csv", index=False)


def compute_partition_metrics(
    partitions: OrderedDict[float, np.ndarray],
    labels_true: np.ndarray,
) -> dict[float, dict[str, float]]:
    """Compute NMI, ARI, AMI, BCubed-F, V-measure, Silhouette per cutoff."""
    from sklearn.metrics import (
        normalized_mutual_info_score,
        adjusted_rand_score,
        adjusted_mutual_info_score,
        v_measure_score,
    )

    results = {}
    for th, pred in partitions.items():
        row: dict[str, float] = {
            "NMI": float(normalized_mutual_info_score(labels_true, pred)),
            "ARI": float(adjusted_rand_score(labels_true, pred)),
            "AMI": float(adjusted_mutual_info_score(labels_true, pred)),
            "V_measure": float(v_measure_score(labels_true, pred)),
            "BCubed_F": float(_bcubed_f(labels_true, pred)),
        }
        results[th] = row
    return results


def _bcubed_f(true: np.ndarray, pred: np.ndarray) -> float:
    """BCubed F-score."""
    n = len(true)
    precision_sum = recall_sum = 0.0
    for i in range(n):
        same_cluster = pred == pred[i]
        same_class = true == true[i]
        cluster_size = same_cluster.sum()
        class_size = same_class.sum()
        correct = (same_cluster & same_class).sum()
        precision_sum += correct / cluster_size
        recall_sum += correct / class_size
    prec = precision_sum / n
    rec = recall_sum / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def two_stage_threshold_scan(
    Z: np.ndarray,
    labels_true: Optional[np.ndarray] = None,
    coarse_min: float = 0.05,
    coarse_max: float = 1.0,
    coarse_step: float = 0.05,
    fine_step: float = 0.01,
    window_expand: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-stage scan returning (coarse_cutoffs, fine_cutoffs)."""
    coarse = np.arange(coarse_min, coarse_max + 1e-9, coarse_step)
    coarse = np.round(coarse, 6)

    if labels_true is None:
        return coarse, coarse

    parts_coarse = build_partitions_from_linkage(Z, coarse.tolist())
    metrics = compute_partition_metrics(parts_coarse, labels_true)

    metric_keys = ["NMI", "ARI", "AMI", "BCubed_F", "V_measure"]
    optima = []
    for key in metric_keys:
        best_th = max(metrics.keys(), key=lambda t: metrics[t].get(key, 0.0))
        optima.append(best_th)

    window_low = max(coarse_min, min(optima) - window_expand)
    window_high = min(coarse_max, max(optima) + window_expand)
    fine = np.arange(window_low, window_high + 1e-9, fine_step)
    fine = np.round(fine, 6)
    return coarse, fine
```

- [ ] **Step 4: Implement tree visualisation (partition panel)**

Add `plot_upgma_partition_tree_panel()` to `partition.py`. Port directly from `ref/embeddings_tree20260206.py` lines ~1117–1394 (the UPGMA drawing code). This function renders the dendrogram + coloured partition bars as a PDF.

The function signature to implement (used by the cluster CLI in Task 8 Step 5):

```python
def plot_upgma_partition_tree_panel(
    Z: np.ndarray,               # scipy linkage matrix from build_upgma()
    ids: list[str],              # sample IDs in linkage order
    partitions: OrderedDict,     # {threshold: label_array} from build_partitions_from_linkage()
    out_path: Path,              # output PDF path
    tree_label: str = "Distance",
    title: str = "",
    palette: str = "tab20",
    bar_width: float = 0.75,
    support_dict: Optional[dict] = None,   # from compute_bootstrap_support()
    bootstrap_cutoff: float = 50.0,
) -> list[int]:                  # returns leaf_order (indices into ids)
    ...
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_partition.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/otuformer/delineation/partition.py tests/test_partition.py
git commit -m "feat: add partition scanning, export, metrics, and tree visualisation"
```

---

## Task 6: `delineation/annotate.py`

**Files:**
- Create: `src/otuformer/delineation/annotate.py`
- Create: `tests/test_annotate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_annotate.py
import pandas as pd
from otuformer.delineation.annotate import apply_corrections


def test_corrections_override_assignments():
    assignments = pd.DataFrame({
        "id": ["img1", "img2", "img3"],
        "cluster": ["OTU_1", "OTU_1", "OTU_2"],
    })
    corrections = pd.DataFrame({
        "id": ["img2"],
        "corrected_cluster": ["OTU_3"],
    })
    result = apply_corrections(assignments, corrections)
    assert result.loc[result["id"] == "img1", "cluster"].iloc[0] == "OTU_1"
    assert result.loc[result["id"] == "img2", "cluster"].iloc[0] == "OTU_3"
    assert result.loc[result["id"] == "img3", "cluster"].iloc[0] == "OTU_2"


def test_uncorrected_ids_unchanged():
    assignments = pd.DataFrame({
        "id": ["a", "b"],
        "cluster": ["OTU_1", "OTU_2"],
    })
    corrections = pd.DataFrame({"id": [], "corrected_cluster": []})
    result = apply_corrections(assignments, corrections)
    assert list(result["cluster"]) == ["OTU_1", "OTU_2"]


def test_annotation_summary_counts():
    from otuformer.delineation.annotate import build_annotation_summary
    assignments = pd.DataFrame({
        "id": ["a", "b", "c"],
        "cluster": ["OTU_1", "OTU_1", "OTU_2"],
    })
    corrections = pd.DataFrame({
        "id": ["b"],
        "corrected_cluster": ["OTU_3"],
    })
    result = apply_corrections(assignments, corrections)
    summary = build_annotation_summary(assignments, result)
    assert summary["n_corrections"] == 1
    assert summary["n_clusters_affected"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_annotate.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `delineation/annotate.py`**

```python
# src/otuformer/delineation/annotate.py
"""Expert taxonomic correction write-back for partition assignments."""

from __future__ import annotations

import pandas as pd


def apply_corrections(
    assignments: pd.DataFrame,
    corrections: pd.DataFrame,
) -> pd.DataFrame:
    """Overwrite cluster assignments with expert corrections.

    Args:
        assignments: DataFrame with 'id' and 'cluster' columns.
        corrections: DataFrame with 'id' and 'corrected_cluster' columns.

    Returns:
        Updated DataFrame with corrections applied.
    """
    result = assignments.copy()
    if len(corrections) == 0:
        return result
    corr_map = dict(zip(corrections["id"].astype(str), corrections["corrected_cluster"]))
    mask = result["id"].astype(str).isin(corr_map)
    result.loc[mask, "cluster"] = result.loc[mask, "id"].astype(str).map(corr_map)
    return result


def build_annotation_summary(
    original: pd.DataFrame,
    corrected: pd.DataFrame,
) -> dict:
    """Build summary statistics for the annotation step."""
    changed = original["cluster"].values != corrected["cluster"].values
    n_corrections = int(changed.sum())
    affected_original = set(original.loc[changed, "cluster"])
    affected_corrected = set(corrected.loc[changed, "cluster"])
    return {
        "n_corrections": n_corrections,
        "n_clusters_affected": len(affected_original | affected_corrected),
        "original_cluster_distribution": original["cluster"].value_counts().to_dict(),
        "corrected_cluster_distribution": corrected["cluster"].value_counts().to_dict(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_annotate.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/annotate.py tests/test_annotate.py
git commit -m "feat: add expert annotation write-back"
```

---

## Task 7: `delineation/diversity.py` — alpha diversity + MPD

**Files:**
- Create: `src/otuformer/delineation/diversity.py`
- Create: `tests/test_diversity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_diversity.py
import numpy as np
import pandas as pd
import pytest
from otuformer.delineation.diversity import (
    compute_alpha_diversity,
    filter_by_min_abundance,
)


def sample_assignments():
    return pd.DataFrame({
        "id": [f"img{i}" for i in range(20)],
        "cluster": ["OTU_1"] * 10 + ["OTU_2"] * 5 + ["OTU_3"] * 3 + ["OTU_4"] * 2,
    })


def test_alpha_diversity_returns_all_indices():
    df = sample_assignments()
    result = compute_alpha_diversity(df)
    expected_keys = ["Richness", "Shannon", "Simpson", "InverseSimpson",
                     "Pielou_J", "Chao1", "Berger_Parker", "Hill_q0", "Hill_q1", "Hill_q2"]
    for k in expected_keys:
        assert k in result, f"Missing key: {k}"


def test_richness_correct():
    df = sample_assignments()
    result = compute_alpha_diversity(df)
    assert result["Richness"] == 4


def test_filter_by_min_abundance():
    df = sample_assignments()
    filtered = filter_by_min_abundance(df, min_abundance=3)
    # OTU_4 has 2 members → filtered out
    assert "OTU_4" not in filtered["cluster"].values
    assert "OTU_1" in filtered["cluster"].values


def test_filter_zero_returns_all():
    df = sample_assignments()
    filtered = filter_by_min_abundance(df, min_abundance=0)
    assert len(filtered) == len(df)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_diversity.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `delineation/diversity.py`**

All computations in pure Python (numpy/scipy/scikit-bio). No usearch.

```python
# src/otuformer/delineation/diversity.py
"""Alpha diversity indices computed from partition assignments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def filter_by_min_abundance(
    assignments: pd.DataFrame, min_abundance: int
) -> pd.DataFrame:
    """Remove clusters with fewer than min_abundance members."""
    if min_abundance <= 0:
        return assignments.copy()
    counts = assignments["cluster"].value_counts()
    keep = counts[counts >= min_abundance].index
    return assignments[assignments["cluster"].isin(keep)].copy()


def _abundance_vector(assignments: pd.DataFrame) -> np.ndarray:
    """Return sorted cluster abundance counts."""
    return np.array(sorted(assignments["cluster"].value_counts().values, reverse=True))


def compute_alpha_diversity(assignments: pd.DataFrame) -> dict[str, float]:
    """Compute all alpha diversity indices from cluster assignments.

    Returns dict of index_name → float value.
    """
    counts = _abundance_vector(assignments)
    n = counts.sum()
    S = len(counts)  # richness
    props = counts / n

    # Shannon H'
    shannon = -float(np.sum(props * np.log(props + 1e-300)))

    # Simpson 1-D
    simpson = float(1.0 - np.sum(props ** 2))

    # Inverse Simpson
    inv_simpson = float(1.0 / np.sum(props ** 2)) if np.sum(props ** 2) > 0 else 0.0

    # Pielou's J
    pielou_j = shannon / math.log(S) if S > 1 else 0.0

    # Heip's E
    heip_e = (math.exp(shannon) - 1) / (S - 1) if S > 1 else 0.0

    # Berger-Parker
    berger_parker = float(counts[0] / n)

    # Margalef
    margalef = (S - 1) / math.log(n) if n > 1 else 0.0

    # Menhinick
    menhinick = S / math.sqrt(n) if n > 0 else 0.0

    # Chao1 (non-parametric richness estimator)
    n1 = int((counts == 1).sum())  # singletons
    n2 = int((counts == 2).sum())  # doubletons
    chao1 = float(S + (n1 * (n1 - 1)) / (2 * (n2 + 1))) if n2 > 0 else float(S + n1 * (n1 - 1) / 2)

    # ACE (Abundance-based Coverage Estimator)
    rare = counts[counts <= 10]
    S_rare = len(rare)
    S_abund = S - S_rare
    n_rare = rare.sum()
    f1 = int((counts == 1).sum())
    if n_rare > 0 and S_rare > 0:
        C_ace = 1 - f1 / n_rare
        if C_ace > 0:
            gamma_sq = max(0, S_rare / C_ace * sum(k * (k - 1) * (counts == k).sum() for k in range(1, 11)) / (n_rare * (n_rare - 1)) - 1)
            ace = float(S_abund + S_rare / C_ace + f1 * gamma_sq / C_ace)
        else:
            ace = float(S)
    else:
        ace = float(S)

    # Brillouin
    brillouin = (math.lgamma(n + 1) - sum(math.lgamma(c + 1) for c in counts)) / n if n > 0 else 0.0

    # Fisher's alpha (solve S = alpha * ln(1 + n/alpha))
    alpha_fisher = _fisher_alpha(S, n)

    # Hill numbers
    hill_q0 = float(S)
    hill_q1 = float(math.exp(shannon))
    hill_q2 = float(inv_simpson)

    return {
        "Richness": float(S),
        "Chao1": chao1,
        "ACE": ace,
        "Margalef": margalef,
        "Menhinick": menhinick,
        "Shannon": shannon,
        "Simpson": simpson,
        "InverseSimpson": inv_simpson,
        "Brillouin": brillouin,
        "Fisher_alpha": alpha_fisher,
        "Pielou_J": pielou_j,
        "Heip_E": heip_e,
        "Berger_Parker": berger_parker,
        "Hill_q0": hill_q0,
        "Hill_q1": hill_q1,
        "Hill_q2": hill_q2,
    }


def _fisher_alpha(S: int, n: int, max_iter: int = 100, tol: float = 1e-6) -> float:
    """Estimate Fisher's alpha by Newton-Raphson."""
    if n <= 0 or S <= 0:
        return 0.0
    alpha = S / math.log(1 + n)
    for _ in range(max_iter):
        x = n / (n + alpha)
        f = alpha * math.log(1 + n / alpha) - S
        df = math.log(1 + n / alpha) - n / (n + alpha)
        if abs(df) < 1e-12:
            break
        alpha_new = alpha - f / df
        if alpha_new <= 0:
            alpha_new = alpha / 2
        if abs(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new
    return float(alpha)


def compute_mpd(
    assignments: pd.DataFrame,
    tree_newick_path: Path,
) -> float:
    """Compute MPD (Morphological Dendrogram Diversity) using scikit-bio Faith PD.

    NOTE: This is MPD (morphological dendrogram diversity), NOT true phylogenetic diversity.
    The UPGMA tree encodes morphological distances, not evolutionary history.
    """
    try:
        from skbio import TreeNode
        from skbio.diversity import alpha_diversity
    except ImportError:
        raise ImportError("scikit-bio required for MPD computation: pip install scikit-bio")

    tree = TreeNode.read(str(tree_newick_path))
    cluster_counts = assignments["cluster"].value_counts()

    # Build OTU table: rows = samples (single sample), cols = OTU tips
    otu_ids = list(cluster_counts.index)
    counts_arr = np.array([cluster_counts.get(o, 0) for o in otu_ids], dtype=int)

    # Faith PD requires tree tips to match OTU ids
    # Prune tree to OTU tips only
    try:
        tree_pruned = tree.shear(otu_ids)
        result = alpha_diversity("faith_pd", counts_arr.reshape(1, -1), otu_ids, tree=tree_pruned)
        return float(result[0])
    except Exception:
        return float("nan")


def diversity_table(
    assignments: pd.DataFrame,
    min_abundances: list[int],
    tree_newick_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute diversity indices for each min_abundance threshold.

    Returns wide DataFrame: rows = index names, columns = min_abundance_N.
    """
    records: dict[str, dict[str, float]] = {}

    for min_ab in min_abundances:
        filtered = filter_by_min_abundance(assignments, min_ab)
        col = f"min_abundance_{min_ab}"
        if len(filtered) == 0:
            records[col] = {}
            continue
        metrics = compute_alpha_diversity(filtered)
        if tree_newick_path is not None:
            try:
                metrics["MPD"] = compute_mpd(filtered, tree_newick_path)
            except Exception:
                metrics["MPD"] = float("nan")
        records[col] = metrics

    df = pd.DataFrame(records)
    df.index.name = "index"
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_diversity.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/otuformer/delineation/diversity.py tests/test_diversity.py
git commit -m "feat: add alpha diversity indices and MPD computation"
```

---

## Task 8: Wire CLI commands

**Files:**
- Modify: `src/otuformer/cli/extract.py`
- Modify: `src/otuformer/cli/evaluate.py`
- Modify: `src/otuformer/cli/cluster.py`
- Modify: `src/otuformer/cli/annotate.py`
- Modify: `src/otuformer/cli/diversity.py`
- Modify: `tests/test_cli_smoke.py`

For each command, replace the stub body with a thin service call. Pattern (shown for `annotate` as simplest example):

- [ ] **Step 1: Wire `cli/annotate.py`**

```python
# replace stub body in annotate callback:
from otuformer.delineation.annotate import apply_corrections, build_annotation_summary
from otuformer.utils.io import read_csv, write_csv, write_json

assignments_df = read_csv(assignments)
corrections_df = read_csv(corrections)
result_df = apply_corrections(assignments_df, corrections_df)
summary = build_annotation_summary(assignments_df, result_df)

out_dir.mkdir(parents=True, exist_ok=True)
stem = assignments.stem
write_csv(result_df, out_dir / f"{stem}_annotated.csv")
write_json(summary, out_dir / "annotation_summary.json")
typer.echo(f"Corrections applied: {summary['n_corrections']}")
typer.echo(f"Clusters affected: {summary['n_clusters_affected']}")
typer.echo(f"Output: {out_dir / f'{stem}_annotated.csv'}")
```

- [ ] **Step 2: Wire `cli/diversity.py`**

Parse `--min-abundance` string → list of ints. Call `diversity_table()`. Write CSV + simple bar chart PDF.

- [ ] **Step 3: Wire `cli/extract.py`**

Call `extract_embeddings()`. Write `embeddings.csv` to `--out-dir`.

- [ ] **Step 4: Wire `cli/evaluate.py`**

Load embeddings CSV + labels CSV. Call evaluation functions. Write `metrics.json`, `metrics.csv`, `umap.pdf`.

- [ ] **Step 5: Wire `cli/cluster.py`**

Full pipeline:
1. Load embeddings CSV
2. Optional PCA whitening
3. Compute distance matrix (cosine or euclidean)
4. Optional local scaling
5. `build_upgma()` → `upgma_to_newick()` → save `upgma.nwk`
6. Plot distance distributions
7. `two_stage_threshold_scan()` (or use `--custom-cutoffs`)
8. `build_partitions_from_linkage()` → `export_partition_tables()`
9. `plot_upgma_partition_tree_panel()` per cutoff
10. If `--labels`: `compute_partition_metrics()` → save metrics CSV

- [ ] **Step 6: Add smoke tests**

```python
# append to tests/test_cli_smoke.py

def test_annotate_command(tmp_path):
    import pandas as pd
    from otuformer.utils.io import write_csv
    assign = pd.DataFrame({"id": ["a","b","c"], "cluster": ["OTU_1","OTU_1","OTU_2"]})
    corr = pd.DataFrame({"id": ["b"], "corrected_cluster": ["OTU_3"]})
    write_csv(assign, tmp_path / "assignments.csv")
    write_csv(corr, tmp_path / "corrections.csv")
    result = runner.invoke(app, [
        "annotate",
        "--assignments", str(tmp_path / "assignments.csv"),
        "--corrections", str(tmp_path / "corrections.csv"),
        "--out-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 0
    assert (tmp_path / "out" / "assignments_annotated.csv").exists()
```

- [ ] **Step 7: Run all tests**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/otuformer/cli/ tests/test_cli_smoke.py
git commit -m "feat: wire extract/evaluate/cluster/annotate/diversity CLI commands"
```

---

## Task 9: Final integration check

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 2: Manual CLI checks**

```bash
otuformer extract --help
otuformer cluster --help
otuformer diversity --help
otuformer annotate --help
```

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: plan 3 complete — embedding and delineation modules"
```
