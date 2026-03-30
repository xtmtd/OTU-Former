#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding toolkit: PCA whitening → cosine distance → local scaling → tree construction

Workflow:
    1. Load train/test embeddings
    2. Optional PCA whitening (fitted on train or combined data)
    3. Compute cosine distances
    4. Optional local scaling (Mutual Proximity)
    5. Build NJ/UPGMA trees with partition analysis
"""

import argparse
import io
import os
import sys
import json
import gc
from pathlib import Path
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from sklearn.decomposition import PCA

from skbio import DistanceMatrix
from skbio.tree import nj

sns.set_style("whitegrid")


# ==============================================================================
# CLI Arguments and Log
# ==============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCA whitening → cosine distance → local scaling → tree construction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # === Input/Output ===
    parser.add_argument("--embeddings_train_csv", required=False,
                        help="CSV with training embeddings (1st column = ID).")
    parser.add_argument("--embeddings_test_csv", default=None,
                        help="CSV with test embeddings (1st column = ID). Optional.")
    parser.add_argument("--tree_data", default="test_only",  # 改为 test_only 作为默认值
                        choices=["test", "train", "train_test", "test_only"],
                        help="Which embeddings to use for tree building: "
                             "'test' → only test CSV (with train-fitted PCA); "
                             "'train' → only train CSV; "
                             "'train_test' → concatenated train then test; "
                             "'test_only' → only test CSV (independent, no train data needed).")
    parser.add_argument("--out_dir", default="results",
                        help="Base output directory.")

    # === Label Data for Metrics ===
    parser.add_argument("--assess_partition_metrics", action="store_true",
                        help="Enable partition quality metrics assessment.")
    parser.add_argument("--label_data_csv", default=None,
                        help="CSV file with labels (columns: ID and 'label'). "
                             "Must correspond to --tree_data samples.")
    parser.add_argument("--metrics_sample_size", type=int, default=10000,
                        help="Sample size for metrics computation (0=use all data, default=10000 for efficiency)")

    # === PCA Whitening ===
    parser.add_argument("--pca_whitening", action="store_true",
                        help="Enable PCA whitening (dimensionality reduction).")
    parser.add_argument("--pca_components", type=int, default=256,
                        help="Number of PCA components to retain (if --pca_whitening enabled).")
    
    # === Local Scaling ===
    parser.add_argument("--enable_local_scaling", action="store_true",
                        help="Enable local scaling (Mutual Proximity). Default: disabled (use raw cosine).")
    parser.add_argument("--local_k", type=int, default=0,
                        help="Number of nearest neighbors for local scaling (0 = auto).")
    parser.add_argument("--local_k_strategy", default="adaptive",
                        choices=["adaptive", "sqrt", "log", "fixed"],
                        help="Strategy for auto k selection when --local_k=0.")
    
    # === Distance Cutoff ===
    parser.add_argument("--distance_cutoff_min", type=float, default=0.4,
                        help="Minimum distance cutoff for partition scan.")
    parser.add_argument("--distance_cutoff_max", type=float, default=0.7,
                        help="Maximum distance cutoff for partition scan.")
    parser.add_argument("--distance_cutoff_step", type=float, default=0.02,
                        help="Step size for distance cutoff scan.")
    parser.add_argument("--custom_cutoffs", type=str, default=None,
                        help="Comma-separated custom cutoff values (overrides min/max/step settings). "
                             "Example: --custom_cutoffs 0.3,0.4,0.5,0.6")
    
    # === Tree Options ===
    parser.add_argument("--UPGMA", action="store_true",
                        help="Enable UPGMA tree construction (default: enabled).")
    parser.add_argument("--NJ", action="store_true",
                        help="Enable Neighbor-Joining tree construction.")
    
    # === Bootstrap ===
    parser.add_argument("--num_bootstraps", type=int, default=0,
                        help="Number of bootstrap replicates (0 = disable).")
    parser.add_argument("--bootstrap_subsample_ratio", type=float, default=0.8,
                        help="Fraction of embedding dimensions to resample in bootstrap.")
    parser.add_argument("--bootstrap_display_cutoff", type=float, default=50.0,
                        help="Minimum bootstrap support value to display on tree.")
    parser.add_argument("--save_bootstrap_trees", action="store_true",
                        help="Save all bootstrap replicate trees to file.")
    
    # === Misc ===
    parser.add_argument("--save_pairwise_distances", action="store_true",
                        help="Save pairwise distance matrices to CSV.")
    parser.add_argument("--max_distance_pairs", type=int, default=1_000_000,
                        help="Max pairs to sample for distance histograms.")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--cpus", type=int, default=8,
                        help="Number of CPU threads for computation (PyTorch/MKL) and bootstrap.")
    
    args = parser.parse_args()
    
    if args.tree_data == "test_only":
        if not args.embeddings_test_csv:
            parser.error("--tree_data=test_only requires --embeddings_test_csv")
    else:
        if not args.embeddings_train_csv:
            parser.error(f"--tree_data={args.tree_data} requires --embeddings_train_csv")
        
        if args.tree_data in ["test", "train_test"] and not args.embeddings_test_csv:
            parser.error(f"--tree_data={args.tree_data} requires --embeddings_test_csv")
    
    if args.assess_partition_metrics and args.label_data_csv:
        if not os.path.exists(args.label_data_csv):
            parser.error(f"Label data file not found: {args.label_data_csv}")    

    return args


class TeeLogger:
    """Redirect stdout to both console and file (skip progress bars)."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = log_path.open('w', encoding='utf-8')
        self.skip_patterns = [
            '\r',
            '|<E2><96><88>', '|<E2><96><91>',
            '\x1b[',
            '[A',
        ]
        self.last_line_was_progress = False
        
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        is_progress = any(pattern in message for pattern in self.skip_patterns)
        if not is_progress:
            if self.last_line_was_progress and message.strip():
                self.log.write('\n')
            self.log.write(message)
            self.log.flush()
            self.last_line_was_progress = False
        else:
            self.last_line_was_progress = True

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# ==============================================================================
# Filesystem Utilities
# ==============================================================================

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def format_threshold(th):
    """Format threshold for filename (remove trailing zeros)."""
    return f"{th:.4f}".rstrip("0").rstrip(".")


# ==============================================================================
# Data Loading
# ==============================================================================

def load_embeddings(csv_path):
    """Load embeddings from CSV (1st column = ID, rest = features)."""
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("Embeddings CSV must contain ID column + feature columns.")
    df = df.set_index(df.columns[0])
    return df


def prepare_embeddings(df):
    """Extract numpy array and ID list from DataFrame."""
    X = df.to_numpy(dtype=np.float64)
    ids = df.index.astype(str).tolist()
    return X, ids


def remove_or_fix_zero_vectors(X, ids, eps=1e-12):
    """Replace zero vectors with minimal non-zero vector to avoid NaN in normalization."""
    norms = np.linalg.norm(X, axis=1)
    zero_mask = norms < eps
    if zero_mask.any():
        print(f"  [WARNING] Found {zero_mask.sum()} zero vectors; replacing with [1,0,...]")
        X = X.copy()
        X[zero_mask] = 0.0
        X[zero_mask, 0] = 1.0
    return X, ids


def load_label_data(csv_path, ids_tree):
    """
    Load label data and align with tree IDs.
    Performs validation to ensure IDs match.
    
    Args:
        csv_path: Path to CSV with columns [ID, label]
        ids_tree: List of IDs used in tree construction
    
    Returns:
        labels: Array of labels aligned with ids_tree (or None if no labels)
        id_to_label: Dict mapping ID to label
    """
    if csv_path is None:
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        
        if 'label' not in df.columns:
            print(f"  [ERROR] 'label' column not found in {csv_path}")
            print(f"  Available columns: {list(df.columns)}")
            return None, None
        
        id_col = df.columns[0]
        label_ids = set(df[id_col].astype(str))
        tree_ids = set(ids_tree)
        
        # Check ID consistency
        print(f"  Label file contains {len(label_ids)} unique IDs")
        print(f"  Tree data contains {len(tree_ids)} unique IDs")
        
        # IDs in tree but not in labels
        missing_in_labels = tree_ids - label_ids
        if missing_in_labels:
            print(f"  [WARNING] {len(missing_in_labels)} tree IDs not found in label file")
            if len(missing_in_labels) <= 10:
                print(f"  Missing in labels: {sorted(list(missing_in_labels))}")
            else:
                print(f"  First 10 missing: {sorted(list(missing_in_labels))[:10]}")
        
        # IDs in labels but not in tree
        extra_in_labels = label_ids - tree_ids
        if extra_in_labels:
            print(f"  [WARNING] {len(extra_in_labels)} label IDs not found in tree data")
            if len(extra_in_labels) <= 10:
                print(f"  Extra in labels: {sorted(list(extra_in_labels))}")
            else:
                print(f"  First 10 extra: {sorted(list(extra_in_labels))[:10]}")
        
        # Check for complete mismatch
        overlap = len(tree_ids & label_ids)
        if overlap == 0:
            print(f"  [ERROR] No overlapping IDs between tree and label data!")
            print(f"  Sample tree IDs: {list(tree_ids)[:5]}")
            print(f"  Sample label IDs: {list(label_ids)[:5]}")
            return None, None
        
        overlap_ratio = overlap / len(tree_ids)
        print(f"  ID overlap: {overlap}/{len(tree_ids)} ({overlap_ratio*100:.1f}%)")
        
        if overlap_ratio < 0.5:
            print(f"  [WARNING] Less than 50% ID overlap - check ID format consistency!")
        
        # Create ID to label mapping
        id_to_label = dict(zip(df[id_col].astype(str), df['label']))
        
        # Align labels with tree IDs
        labels = []
        for tree_id in ids_tree:
            if tree_id in id_to_label:
                labels.append(id_to_label[tree_id])
            else:
                labels.append(None)
        
        labels = np.array(labels)
        valid_labels = labels[labels != None]
        
        if len(valid_labels) == 0:
            print(f"  [ERROR] No valid labels found after alignment")
            return None, None
        
        unique_labels = np.unique(valid_labels)
        print(f"  Successfully loaded labels for {len(valid_labels)}/{len(ids_tree)} samples")
        print(f"  Unique labels: {len(unique_labels)}")
        
        # Show label distribution
        if len(unique_labels) <= 20:
            from collections import Counter
            label_counts = Counter(valid_labels)
            print(f"  Label distribution:")
            for label, count in sorted(label_counts.items()):
                print(f"    {label}: {count}")
        
        return labels, id_to_label
        
    except Exception as e:
        print(f"  [ERROR] Failed to load label data: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ==============================================================================
# PCA Whitening
# ==============================================================================

def apply_pca_whitening(X_train, X_test, n_components, random_state=42):
    """
    Apply PCA whitening to training and test data.
    
    PCA is fitted on training data only, then applied to both train and test.
    
    Args:
        X_train: Training embeddings (n_train, d)
        X_test: Test embeddings (n_test, d) or None
        n_components: Number of PCA components to retain
        random_state: Random seed for PCA
    
    Returns:
        X_train_pca: Whitened training embeddings
        X_test_pca: Whitened test embeddings (or None if X_test is None)
        pca: Fitted PCA object
    """
    print(f"  Applying PCA whitening: {X_train.shape[1]} → {n_components} dimensions")
    
    # Fit PCA on training data with whitening
    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance: {explained_var:.4f}")
    
    # Transform test data if provided
    X_test_pca = None
    if X_test is not None:
        X_test_pca = pca.transform(X_test)
    
    return X_train_pca, X_test_pca, pca


# ==============================================================================
# Distance Computation
# ==============================================================================

def safe_l2_normalize(X, eps=1e-12):
    """L2 normalize rows, avoiding division by zero."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def compute_cosine_distances(X, chunk_size=None, use_triangular=None):
    """
    Compute pairwise cosine distances with automatic optimization.
    
    Automatically selects optimal strategy:
    - Small data (n < 10k): Standard float64 computation
    - Large data (n >= 10k): Chunked triangular float32→float64 computation
    
    Args:
        X: Embeddings (n_samples, n_features)
        chunk_size: Chunk size for large datasets (None = auto: min(5000, n//20))
        use_triangular: Force strategy (None = auto based on size)
    
    Returns:
        distance: Square distance matrix (n, n), float64
    """
    n = X.shape[0]
    
    # Auto-detect optimization strategy
    if use_triangular is None:
        use_triangular = (n >= 10000)
    
    # === Small dataset: Standard method ===
    if not use_triangular:
        print(f"  Computing distances [standard]: n={n}")
        X_norm = safe_l2_normalize(X)
        similarity = X_norm @ X_norm.T
        distance = 1.0 - similarity
        np.fill_diagonal(distance, 0.0)
        distance = (distance + distance.T) / 2.0
        return np.clip(distance, 0.0, 2.0)
    
    # === Large dataset: Optimized triangular method ===
    if chunk_size is None:
        # Auto-select: balance between memory and I/O
        chunk_size = min(5000, max(1000, n // 20))
    
    #print(f"  Computing distances [triangular+chunked]: n={n}, chunk={chunk_size}")
    #print(f"    Strategy: float32 computation → float64 output")
    #print(f"    Memory: ~{n*n*4/(1024**3):.1f}GB peak (vs {n*n*8/(1024**3):.1f}GB standard)")
    
    X_norm = safe_l2_normalize(X.astype(np.float32))
    distance = np.zeros((n, n), dtype=np.float32)
    n_chunks = (n + chunk_size - 1) // chunk_size
    total_blocks = n_chunks * (n_chunks + 1) // 2
    computed_blocks = 0
    
    for i in range(n_chunks):
        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, n)
        
        for j in range(i, n_chunks):
            start_j = j * chunk_size
            end_j = min((j + 1) * chunk_size, n)
            
            similarity_block = X_norm[start_i:end_i] @ X_norm[start_j:end_j].T
            distance_block = 1.0 - similarity_block
            distance[start_i:end_i, start_j:end_j] = distance_block
            
            if i != j:
                distance[start_j:end_j, start_i:end_i] = distance_block.T
            
            computed_blocks += 1
            if computed_blocks % 10 == 0 or computed_blocks == total_blocks:
                progress = computed_blocks * 100 // total_blocks
                mem_gb = distance.nbytes / (1024**3)
                #print(f"    Block {computed_blocks}/{total_blocks} ({progress}%) | Matrix: {mem_gb:.1f}GB")
    
    # Post-processing
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0  # Ensure exact symmetry
    distance = np.clip(distance, 0.0, 2.0)

    if not np.isfinite(distance).all():
        raise ValueError("Distance matrix contains non-finite values (NaN or Inf)")

    return distance


def validate_distance_matrix(square, atol=1e-8):
    """Ensure distance matrix is symmetric with zero diagonal."""
    square = np.asarray(square)
    
    if not np.isfinite(square).all():
        raise ValueError("Distance matrix contains non-finite entries.")
    
    # Ensure symmetry
    if not np.allclose(square, square.T, atol=atol):
        square = (square + square.T) / 2.0
    
    # Ensure zero diagonal
    np.fill_diagonal(square, 0.0)
    
    return square


# ==============================================================================
# Local Scaling (Mutual Proximity)
# ==============================================================================

def auto_select_k(n_samples, strategy='adaptive'):
    """
    Automatically select k for local scaling based on dataset size.
    
    Args:
        n_samples: Number of samples
        strategy: Selection strategy
            - 'fixed': Always use k=7 (classic default)
            - 'sqrt': k = sqrt(n)
            - 'log': k = 5 * log2(n)
            - 'adaptive': Hybrid approach with bounds
    
    Returns:
        k: Number of nearest neighbors
    """
    if strategy == 'fixed':
        k = 7
    
    elif strategy == 'sqrt':
        k = int(np.sqrt(n_samples))
    
    elif strategy == 'log':
        k = max(5, int(5 * np.log2(n_samples)))
    
    elif strategy == 'adaptive':
        if n_samples < 100:
            k = 7
        elif n_samples < 500:
            k = int(np.sqrt(n_samples))
        else:
            k = int(5 * np.log2(n_samples))
        
        # Apply bounds: k ∈ [5, 100]
        k = max(5, min(k, 100))
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Ensure k < n_samples (required for kth nearest neighbor)
    k = min(k, n_samples - 1)
    
    return k


def apply_local_scaling(dist_matrix, k):
    """
    Apply Mutual Proximity local scaling to distance matrix.
    
    Formula:
        d_scaled(i,j) = d(i,j) / sqrt(σ_i * σ_j)
        where σ_i = distance to k-th nearest neighbor of point i
    
    Args:
        dist_matrix: Square distance matrix (n, n)
        k: Number of nearest neighbors for scaling
    
    Returns:
        scaled_dist: Scaled distance matrix (n, n)
        knn_dists: Array of k-th NN distances for each point
    """
    n = dist_matrix.shape[0]
    
    # Get k-th nearest neighbor distance for each point
    # (excluding self, so we need k+1-th smallest value overall)
    # np.partition partitions so that element at index k is in correct position
    knn_dists = np.partition(dist_matrix, k, axis=1)[:, k]
    
    # Create scaling matrix: sqrt(σ_i * σ_j)
    scale_matrix = np.sqrt(np.outer(knn_dists, knn_dists))
    
    # Avoid division by zero
    scale_matrix = np.maximum(scale_matrix, 1e-10)
    
    # Apply Mutual Proximity scaling
    scaled_dist = dist_matrix / scale_matrix
    
    # Ensure diagonal is zero
    np.fill_diagonal(scaled_dist, 0.0)
    
    # Ensure symmetry
    scaled_dist = (scaled_dist + scaled_dist.T) / 2.0
    
    return scaled_dist, knn_dists


# ==============================================================================
# Distance Statistics & Diagnostics
# ==============================================================================

def compute_distance_stats(dist_matrix, name="distance"):
    """
    Compute comprehensive statistics for distance matrix.
    
    Returns:
        stats: Dictionary with statistics
    """
    # Extract upper triangle (exclude diagonal)
    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = {p: float(np.percentile(upper_tri, p)) for p in percentiles}
    
    stats = {
        "name": name,
        "n_samples": dist_matrix.shape[0],
        "min": float(upper_tri.min()),
        "max": float(upper_tri.max()),
        "mean": float(upper_tri.mean()),
        "std": float(upper_tri.std()),
        "median": float(np.median(upper_tri)),
        "percentiles": percentile_values
    }
    
    return stats


def compute_intraclass_distance_statistics(dist_matrix, ids, id_to_label, out_path):
    """
    Compute intra-class pairwise distance statistics for each true label class.
    
    Args:
        dist_matrix: Square distance matrix (n, n)
        ids: List of sample IDs
        id_to_label: Dict mapping ID to label
        out_path: Output CSV path
    """
    
    # Build label array aligned with distance matrix
    labels = []
    valid_indices = []
    for i, sample_id in enumerate(ids):
        if sample_id in id_to_label:
            labels.append(id_to_label[sample_id])
            valid_indices.append(i)
        else:
            labels.append(None)
    
    labels = np.array(labels)
    
    # Filter to valid samples only
    if len(valid_indices) == 0:
        print("  [WARNING] No valid labels found for intra-class statistics")
        return
    
    valid_mask = np.array([i in valid_indices for i in range(len(ids))])
    dist_valid = dist_matrix[np.ix_(valid_mask, valid_mask)]
    labels_valid = labels[valid_mask]
    
    # Compute statistics for each class
    unique_labels = np.unique(labels_valid)
    stats_records = []
    
    for label in unique_labels:
        # Get indices for this class
        class_mask = labels_valid == label
        class_indices = np.where(class_mask)[0]
        n_samples = len(class_indices)
        
        if n_samples < 2:
            # Skip classes with only one sample
            stats_records.append({
                'class': label,
                'n_samples': n_samples,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'q25': np.nan,
                'q75': np.nan
            })
            continue
        
        # Extract intra-class distances (upper triangle only to avoid duplicates)
        class_dist = dist_valid[np.ix_(class_indices, class_indices)]
        intra_distances = class_dist[np.triu_indices_from(class_dist, k=1)]
        
        # Compute statistics
        stats_records.append({
            'class': label,
            'n_samples': n_samples,
            'n_pairs': len(intra_distances),
            'mean': float(np.mean(intra_distances)),
            'median': float(np.median(intra_distances)),
            'std': float(np.std(intra_distances)),
            'min': float(np.min(intra_distances)),
            'max': float(np.max(intra_distances)),
            'q25': float(np.percentile(intra_distances, 25)),
            'q75': float(np.percentile(intra_distances, 75))
        })
    
    # Create DataFrame and save
    stats_df = pd.DataFrame(stats_records)
    stats_df = stats_df.sort_values('class').reset_index(drop=True)
    
    stats_df.to_csv(out_path, index=False)
    print(f"  Saved intra-class distance statistics to {out_path}")
    
    # Print summary
    print(f"  Summary: {len(unique_labels)} classes analyzed")
    if len(stats_df) > 0:
        print(f"  Mean distance range: [{stats_df['mean'].min():.4f}, {stats_df['mean'].max():.4f}]")
        print(f"  Overall mean: {stats_df['mean'].mean():.4f}")
    
    return stats_df


def save_distance_statistics(stats_list, out_path):
    """Save distance statistics to JSON file."""
    with open(out_path, 'w') as f:
        json.dump(stats_list, f, indent=2)
    print(f"  Saved distance statistics to {out_path}")


def plot_distance_distributions(dist_matrix, label, out_dir, metric="cosine",
                                max_pairs=1_000_000, random_state=42):
    """
    Plot distance distribution histograms and CDF.
    
    Args:
        dist_matrix: Square distance matrix
        label: Label for this distance type (e.g., "raw" or "scaled")
        out_dir: Output directory
        metric: Distance metric name (for labeling)
        max_pairs: Maximum pairs to sample for histogram
        random_state: Random seed
    """
    ensure_dir(out_dir)
    
    # Extract upper triangle
    upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    
    # Sample if too many pairs
    if len(upper_tri) > max_pairs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(upper_tri), size=max_pairs, replace=False)
        values = upper_tri[idx]
    else:
        values = upper_tri
    
    # Histogram (linear scale)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(values, bins=60, color="#1f77b4", edgecolor="black", ax=ax)
    ax.set_title(f"Distance distribution ({label})")
    ax.set_xlabel(f"{metric} distance")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"distance_hist_{label}.pdf"))
    plt.close(fig)
    
    # CDF
    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color="#d95f02", linewidth=2)
    ax.set_title(f"Cumulative distribution ({label})")
    ax.set_xlabel(f"{metric} distance")
    ax.set_ylabel("Cumulative frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"distance_cum_{label}.pdf"))
    plt.close(fig)
    
    # Histogram (log scale, positive values only)
    positive = values[values > 0]
    if len(positive) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(positive, bins=60, color="#1f77b4", edgecolor="black", ax=ax)
        ax.set_title(f"Distance distribution ({label}, log scale)")
        ax.set_xlabel(f"{metric} distance")
        ax.set_ylabel("Frequency")
        ax.set_yscale("log", base=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"distance_hist_{label}_log.pdf"))
        plt.close(fig)


def save_distance_matrix(square, ids, out_path):
    """Save distance matrix to CSV."""
    df = pd.DataFrame(square, index=ids, columns=ids)
    df.to_csv(out_path)
    print(f"  Saved distance matrix ({square.shape[0]}x{square.shape[1]}) to {out_path}")


# ==============================================================================
# Cutoff Selection
# ==============================================================================

def determine_cutoffs(args):
    """
    Determine cutoff values based on arguments.
    
    Args:
        args: Command-line arguments
    
    Returns:
        cutoffs: Array of cutoff values
    """
    # Custom cutoffs override default range
    if args.custom_cutoffs:
        cutoffs = np.array([float(x.strip()) for x in args.custom_cutoffs.split(',')])
        print(f"  Using custom cutoffs: {cutoffs}")
    else:
        # Use default range
        cutoffs = np.arange(
            args.distance_cutoff_min,
            args.distance_cutoff_max + 1e-9,
            args.distance_cutoff_step
        )
        cutoffs = np.round(cutoffs, 6)
        print(f"  Using cutoff range: [{args.distance_cutoff_min}, {args.distance_cutoff_max}], step={args.distance_cutoff_step}")
        print(f"  Cutoffs: {cutoffs}")
    
    return cutoffs


# ==============================================================================
# Partition Analysis
# ==============================================================================

def build_partitions_from_linkage(Z, thresholds):
    """Build partitions from hierarchical clustering at different thresholds."""
    partitions = OrderedDict()
    for th in sorted(thresholds):
        labels = hierarchy.fcluster(Z, t=th, criterion='distance')
        partitions[float(th)] = labels
    return partitions


def scan_linkage_thresholds(Z, thresholds, out_csv=None, out_fig=None, label=""):
    """Scan thresholds and plot number of clusters vs threshold."""
    records = []
    for th in thresholds:
        labels = hierarchy.fcluster(Z, t=th, criterion="distance")
        n_clusters = len(np.unique(labels))
        records.append({"threshold": th, "clusters": n_clusters})
    
    df = pd.DataFrame(records)
    
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"  Saved partition scan to {out_csv}")
    
    if out_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["threshold"], df["clusters"], marker="o", color="#1f77b4", linewidth=2)
        ax.set_xlabel("Distance cutoff")
        ax.set_ylabel("Number of clusters")
        if label:
            ax.set_title(f"Partition scan: {label}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_fig, dpi=300)
        plt.close(fig)
        print(f"  Saved partition scan plot to {out_fig}")
    
    return df


def export_partition_tables(ids, partitions, out_dir):
    """Export partition assignments to CSV files."""
    ensure_dir(out_dir)
    
    for th, labels in partitions.items():
        # Group IDs by cluster
        clusters = defaultdict(list)
        for lab, tip in zip(labels, ids):
            clusters[lab].append(tip)
        
        # Sort clusters by size (descending) then by first member
        sorted_clusters = sorted(
            clusters.items(),
            key=lambda x: (-len(x[1]), x[1][0])
        )
        
        # Create summary table
        rows_summary = []
        rows_tip_level = []
        
        for idx, (lab, tips) in enumerate(sorted_clusters, start=1):
            cname = f"cluster{idx}"
            rows_summary.append({
                "cluster": cname,
                "size": len(tips),
                "members": ";".join(tips)
            })
            
            for tip in tips:
                rows_tip_level.append({"id": tip, "cluster": cname})
        
        # Save tables
        th_tag = format_threshold(th)
        
        pd.DataFrame(rows_summary).to_csv(
            os.path.join(out_dir, f"partition_{th_tag}_summary.csv"),
            index=False
        )
        
        pd.DataFrame(rows_tip_level).to_csv(
            os.path.join(out_dir, f"partition_{th_tag}_assignments.csv"),
            index=False
        )


# ==============================================================================
# Tree Drawing Utilities
# ==============================================================================

def clean_tip_name(name):
    """Remove quotes from tip names if present."""
    if name is None:
        return None
    name = str(name)
    if (name.startswith("'") and name.endswith("'")) or \
       (name.startswith('"') and name.endswith('"')):
        name = name[1:-1]
    return name


def _contiguous_runs(indices):
    """Yield contiguous runs of indices."""
    if len(indices) == 0:
        return
    block = [indices[0]]
    for idx in indices[1:]:
        if idx == block[-1] + 1:
            block.append(idx)
        else:
            yield block
            block = [idx]
    yield block


# ==============================================================================
# NJ Tree Layout & Drawing
# ==============================================================================

def compute_nj_tree_layout(tree, ids):
    """
    Compute layout for NJ tree with true branch lengths.
    
    Returns:
        tip_coords: dict {tip_name: (x, y)}
        node_depth: dict {node_id: depth}
        node_y_final: dict {node_id: y_position}
        max_depth: maximum depth
        tips_in_order: ordered tip list
    """
    # Get root
    root = tree
    if hasattr(tree, 'root') and callable(tree.root):
        r = tree.root()
        if r is not None:
            root = r
    
    # Compute depths (x-coordinates)
    node_depth = {}
    
    def compute_depth(node, current_depth):
        node_depth[id(node)] = current_depth
        for child in node.children:
            branch_len = float(child.length) if child.length else 0.0
            compute_depth(child, current_depth + branch_len)
    
    compute_depth(root, 0.0)
    
    # Assign y-coordinates to tips
    y_step = 10
    tips_in_order = list(tree.tips())
    tip_y_final = {}
    
    for i, tip in enumerate(tips_in_order):
        tip_y_final[id(tip)] = (y_step / 2) + y_step * i
    
    # Compute internal node y as mean of children
    node_y_final = {}
    
    def compute_node_y(node):
        if node.is_tip():
            node_y_final[id(node)] = tip_y_final[id(node)]
            return tip_y_final[id(node)]
        child_ys = [compute_node_y(child) for child in node.children]
        y = np.mean(child_ys)
        node_y_final[id(node)] = y
        return y
    
    compute_node_y(root)
    
    # Get max depth for x-axis reversal
    max_depth = max(node_depth.values())
    
    # Collect tip coordinates
    tip_coords = {}
    for tip in tree.tips():
        name = clean_tip_name(tip.name)
        x = max_depth - node_depth[id(tip)]
        y = node_y_final[id(tip)]
        tip_coords[name] = (x, y)
    
    return tip_coords, node_depth, node_y_final, max_depth, tips_in_order


def draw_nj_tree_with_extensions(ax, tree, ids, show_extensions=True):
    """
    Draw NJ tree with true branch lengths and dashed extensions.
    
    Returns:
        tip_y_positions: dict {tip_name: y_position}
        leaf_order: list of tip indices in display order
        max_depth: maximum depth
    """
    tip_coords, node_depth, node_y_final, max_depth, tips_in_order = \
        compute_nj_tree_layout(tree, ids)
    
    # Get root
    root = tree
    if hasattr(tree, 'root') and callable(tree.root):
        r = tree.root()
        if r is not None:
            root = r
    
    # Draw edges
    solid_segments = []
    
    def draw_edges(node):
        x = max_depth - node_depth[id(node)]
        y = node_y_final[id(node)]
        
        for child in node.children:
            child_x = max_depth - node_depth[id(child)]
            child_y = node_y_final[id(child)]
            
            # Horizontal line
            solid_segments.append([(x, child_y), (child_x, child_y)])
            # Vertical line
            solid_segments.append([(x, y), (x, child_y)])
            
            draw_edges(child)
    
    draw_edges(root)
    
    # Draw solid branches
    lc = LineCollection(solid_segments, colors='C0', linewidths=1.0)
    ax.add_collection(lc)
    
    # Draw dashed extensions for tips
    if show_extensions:
        dashed_segments = []
        for tip in tree.tips():
            tip_x = max_depth - node_depth[id(tip)]
            tip_y = node_y_final[id(tip)]
            if tip_x > 1e-9:  # Only draw if not already at x=0
                dashed_segments.append([(tip_x, tip_y), (0, tip_y)])
        
        if dashed_segments:
            lc_dashed = LineCollection(
                dashed_segments,
                colors='gray',
                linewidths=0.8,
                linestyles='dashed'
            )
            ax.add_collection(lc_dashed)
    
    # Set axis limits
    y_step = 10
    all_y = list(node_y_final.values())
    ax.set_xlim(-0.02 * max_depth, max_depth * 1.15)
    ax.set_ylim(max(all_y) + y_step, min(all_y) - y_step)
    
    # Create return values
    id_to_idx = {name: i for i, name in enumerate(ids)}
    ordered_tips = sorted(tips_in_order, key=lambda t: node_y_final[id(t)])
    
    tip_y_positions = {}
    leaf_order = []
    
    for tip in ordered_tips:
        name = clean_tip_name(tip.name)
        tip_y_positions[name] = node_y_final[id(tip)]
        if name in id_to_idx:
            leaf_order.append(id_to_idx[name])
    
    return tip_y_positions, leaf_order, max_depth


def annotate_nj_supports(ax, tree, support_dict, max_depth,
                         min_support=50.0, fmt='{:.0f}', color='darkred'):
    """Annotate NJ tree internal nodes with bootstrap support values."""
    if not support_dict:
        return
    
    # Get root
    root = tree
    if hasattr(tree, 'root') and callable(tree.root):
        r = tree.root()
        if r is not None:
            root = r
    
    # Recompute layout
    _, node_depth, node_y_final, _, tips_in_order = compute_nj_tree_layout(tree, [])
    
    # Get axis range for offset
    x_min, x_max = ax.get_xlim()
    x_offset = 0.01 * (x_max - x_min)
    
    n_tips = len(tips_in_order)
    
    for node in tree.non_tips():
        # Get clade
        node_tips = frozenset(clean_tip_name(tip.name) for tip in node.tips())
        
        # Skip root and trivial clades
        if len(node_tips) <= 1 or len(node_tips) >= n_tips:
            continue
        
        # Get support
        support = support_dict.get(node_tips)
        if support is None or support < min_support:
            continue
        
        # Get position
        x = max_depth - node_depth[id(node)]
        y = node_y_final[id(node)]
        
        # Annotate
        ax.text(x + x_offset, y, fmt.format(support),
                ha='left', va='center', fontsize=7, color=color,
                clip_on=False)


# ==============================================================================
# UPGMA Tree Drawing
# ==============================================================================

def plot_upgma_partition_tree_panel(Z, ids, partitions, out_path,
                                    tree_label='Distance', title='',
                                    palette='tab20', bar_width=0.75,
                                    support_dict=None,
                                    bootstrap_cutoff=50.0):
    """Plot UPGMA tree with partition bars."""
    n_samples = len(ids)
    fig = plt.figure(figsize=(14, max(6, 0.25 * n_samples)))
    
    n_parts = len(partitions) if partitions else 1
    base_partition_width = max(1.2, n_parts * bar_width * 0.35)
    partition_width = base_partition_width * 0.8
    width_ratios = [1.4, partition_width, 2.8]
    
    gs = gridspec.GridSpec(1, 3, width_ratios=width_ratios, wspace=0.05)
    ax_tree = fig.add_subplot(gs[0, 2])
    
    # Draw dendrogram
    dendro = hierarchy.dendrogram(
        Z, labels=ids, orientation='right',
        color_threshold=0, no_labels=True, ax=ax_tree,
        above_threshold_color='C0'
    )
    
    leaf_order = dendro['leaves']
    ordered_ids = [ids[i] for i in leaf_order]
    
    # Extend x-axis for annotations
    x_min, x_max = ax_tree.get_xlim()
    ax_tree.set_xlim(x_min, x_max + 0.08 * (x_max - x_min))
    
    y_step = 10
    y_positions = (y_step / 2) + y_step * np.arange(len(ordered_ids))
    y_lookup = {leaf_order[i]: y_positions[i] for i in range(len(leaf_order))}
    tip_y_positions = {ids[idx]: y_lookup[idx] for idx in leaf_order}
    
    # Add partition panels
    ax_names = fig.add_subplot(gs[0, 0], sharey=ax_tree)
    ax_parts = fig.add_subplot(gs[0, 1], sharey=ax_tree)
    
    if partitions:
        plot_partition_panel(ax_names, ax_parts, ordered_ids, partitions,
                            leaf_order, tip_y_positions,
                            palette=palette, bar_width=bar_width)
    else:
        ax_names.axis('off')
        ax_parts.axis('off')
    
    # Sync y-limits
    ymin = y_positions[-1] + y_step
    ymax = y_positions[0] - y_step
    for ax in (ax_names, ax_parts, ax_tree):
        ax.set_ylim(ymin, ymax)
    
    ax_tree.set_xlabel(tree_label)
    ax_tree.set_yticks([])
    for spine in ax_tree.spines.values():
        spine.set_visible(False)
    
    # Add bootstrap support
    if support_dict:
        clade_heights = compute_clade_heights(Z, ids)
        clade_y = compute_clade_y_positions(Z, ids, leaf_order, y_lookup)
        annotate_supports(ax_tree, support_dict, clade_heights, clade_y,
                         min_support=bootstrap_cutoff)
    
    if title:
        fig.suptitle(title, fontsize=12, y=0.972)
    
    fig.subplots_adjust(top=0.97, right=0.97, left=0.02)
    fig.savefig(out_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    return leaf_order


def compute_clade_heights(Z, ids):
    """Compute heights of all clades in linkage matrix."""
    n = len(ids)
    current_index = n
    clusters = {i: {ids[i]} for i in range(n)}
    heights = {}
    
    for row in Z:
        a, b, height, _ = row
        a, b = int(a), int(b)
        clade = clusters[a] | clusters[b]
        
        if 1 < len(clade) < n:
            heights[frozenset(clade)] = height
        
        clusters[current_index] = clade
        current_index += 1
    
    return heights


def compute_clade_y_positions(Z, ids, leaf_order, y_lookup):
    """Compute y-coordinates of internal nodes."""
    n = len(ids)
    node_y = {i: y_lookup[i] for i in range(n)}
    clusters = {i: {ids[i]} for i in range(n)}
    clade_y = {}
    current = n
    
    for a, b, _, _ in Z:
        a, b = int(a), int(b)
        clade = clusters[a] | clusters[b]
        y = 0.5 * (node_y[a] + node_y[b])
        
        if 1 < len(clade) < n:
            clade_y[frozenset(clade)] = y
        
        node_y[current] = y
        clusters[current] = clade
        current += 1
    
    return clade_y


def annotate_supports(ax, support_dict, clade_heights, clade_y_positions,
                     min_support=50.0, fmt='{:.0f}', color='darkred'):
    """Annotate UPGMA tree with bootstrap support values."""
    if not support_dict:
        return
    
    x_min, x_max = ax.get_xlim()
    x_offset = 0.01 * (x_max - x_min)
    
    for clade, support in support_dict.items():
        if support < min_support:
            continue
        if clade not in clade_heights or clade not in clade_y_positions:
            continue
        
        x = clade_heights[clade]
        y = clade_y_positions[clade]
        
        ax.text(x + x_offset, y, fmt.format(support),
                ha='left', va='center', fontsize=7, color=color,
                clip_on=False)


# ==============================================================================
# NJ Tree Panel
# ==============================================================================

def plot_nj_partition_tree_panel(tree, ids, partitions, out_path,
                                 tree_label='Distance', title='',
                                 palette='tab20', bar_width=0.75,
                                 support_dict=None,
                                 bootstrap_cutoff=50.0):
    """Plot NJ tree with true branch lengths and partition bars."""
    n_samples = len(ids)
    fig = plt.figure(figsize=(14, max(6, 0.25 * n_samples)))
    
    n_parts = len(partitions) if partitions else 1
    base_partition_width = max(1.2, n_parts * bar_width * 0.35)
    partition_width = base_partition_width * 0.8
    width_ratios = [1.4, partition_width, 2.8]
    
    gs = gridspec.GridSpec(1, 3, width_ratios=width_ratios, wspace=0.05)
    ax_tree = fig.add_subplot(gs[0, 2])
    
    # Draw NJ tree
    tip_y_positions, leaf_order, max_depth = draw_nj_tree_with_extensions(
        ax_tree, tree, ids, show_extensions=True
    )
    
    # Get ordered IDs
    ordered_ids = [ids[i] for i in leaf_order]
    
    # Add partition panels
    ax_names = fig.add_subplot(gs[0, 0], sharey=ax_tree)
    ax_parts = fig.add_subplot(gs[0, 1], sharey=ax_tree)
    
    if partitions:
        plot_partition_panel(ax_names, ax_parts, ordered_ids, partitions,
                            leaf_order, tip_y_positions,
                            palette=palette, bar_width=bar_width)
    else:
        ax_names.axis('off')
        ax_parts.axis('off')
    
    # Sync y-limits
    y_vals = list(tip_y_positions.values())
    y_step = 10
    ymin = max(y_vals) + y_step
    ymax = min(y_vals) - y_step
    for ax in (ax_names, ax_parts, ax_tree):
        ax.set_ylim(ymin, ymax)
    
    ax_tree.set_xlabel(tree_label)
    ax_tree.set_yticks([])
    for spine in ax_tree.spines.values():
        spine.set_visible(False)
    
    # Add bootstrap support
    if support_dict:
        annotate_nj_supports(ax_tree, tree, support_dict, max_depth,
                            min_support=bootstrap_cutoff)
    
    if title:
        fig.suptitle(title, fontsize=12, y=0.972)
    
    fig.subplots_adjust(top=0.97, right=0.97, left=0.02)
    fig.savefig(out_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    return leaf_order


def plot_partition_panel(ax_names, ax_parts, ordered_ids, partitions,
                         leaf_order, y_positions_dict,
                         palette='tab20', bar_width=0.75):
    """Plot tip names and partition color bars."""
    n_samples = len(ordered_ids)
    y_positions = [y_positions_dict[name] for name in ordered_ids]
    y_step = (y_positions[1] - y_positions[0]) if n_samples > 1 else 10.0
    top_label_y = min(y_positions) - 0.7 * abs(y_step)
    
    # Names panel
    ax_names.axis('off')
    for tip in ordered_ids:
        ax_names.text(0.98, y_positions_dict[tip], tip,
                     ha='right', va='center', fontsize=9)
    
    # Partitions panel
    n_parts = len(partitions)
    if n_parts == 0:
        return
    
    padding = max(0.05, 0.04 * bar_width)
    ax_parts.set_xlim(-bar_width / 2 - padding,
                     (n_parts - 1) + bar_width / 2 + padding)
    ax_parts.set_xticks([])
    ax_parts.set_yticks([])
    for spine in ax_parts.spines.values():
        spine.set_visible(False)
    
    bar_colors = sns.color_palette(palette, max(20, n_parts * 2))
    
    for col_idx, (th, labels) in enumerate(partitions.items()):
        ordered_labels = np.asarray(labels)[leaf_order]
        cluster_ids = list(OrderedDict.fromkeys(ordered_labels))
        colors = {cid: bar_colors[i % len(bar_colors)] 
                 for i, cid in enumerate(cluster_ids)}
        cluster_sizes = {cid: (ordered_labels == cid).sum() 
                        for cid in cluster_ids}
        
        for cid in cluster_ids:
            idxs = np.where(ordered_labels == cid)[0]
            for block in _contiguous_runs(idxs):
                y0 = y_positions[block[0]] - abs(y_step) / 2.0
                height = abs(y_step) * len(block)
                rect = Rectangle(
                    (col_idx - bar_width / 2, y0),
                    bar_width, height,
                    facecolor=colors[cid],
                    edgecolor='white', linewidth=0.5
                )
                ax_parts.add_patch(rect)
                
                # Add cluster size label
                ax_parts.text(
                    col_idx, y0 + height / 2,
                    str(cluster_sizes[cid]),
                    ha='center', va='center', fontsize=8,
                    color='white' if cluster_sizes[cid] > 1 else 'black'
                )
        
        # Add column header (n_clusters and threshold)
        ax_parts.text(
            col_idx, top_label_y,
            f'{len(np.unique(labels))}\n{th:.3f}',
            ha='center', va='bottom', fontsize=7
        )


# ==============================================================================
# Tree Construction
# ==============================================================================

def build_nj_tree(square_mat, ids, midpoint=True, out_path=None):
    """Build Neighbor-Joining tree from distance matrix."""
    dm = DistanceMatrix(square_mat, ids)
    tree = nj(dm)
    
    if midpoint:
        tree = midpoint_root(tree)
    
    # Clean tip names
    for tip in tree.tips():
        tip.name = clean_tip_name(tip.name)
    
    if out_path:
        tree.write(out_path)
    
    return tree


def midpoint_root(tree):
    """Root tree at midpoint."""
    rooted = tree.copy()
    new_root = rooted.root_at_midpoint()
    return rooted if new_root is None else new_root


def tree_to_linkage(tree, tip_labels):
    """Convert skbio TreeNode to scipy linkage matrix."""
    idx_lookup = {name: i for i, name in enumerate(tip_labels)}
    rows = []
    next_id = len(tip_labels)
    
    root = tree
    if hasattr(tree, 'root') and callable(tree.root):
        root = tree.root()
    if root is None:
        root = tree
    
    def postorder(node):
        nonlocal next_id
        if node.is_tip():
            name = clean_tip_name(node.name)
            idx = idx_lookup[name]
            node._linkage_id = idx
            node._subtree_height = 0.0
            return idx, 0.0, 1
        
        if len(node.children) != 2:
            raise ValueError("tree_to_linkage expects strictly bifurcating tree.")
        
        left, right = node.children
        left_idx, left_height, left_size = postorder(left)
        right_idx, right_height, right_size = postorder(right)
        
        left_branch = float(left.length or 0.0)
        right_branch = float(right.length or 0.0)
        left_total = left_height + left_branch
        right_total = right_height + right_branch
        
        merge_distance = left_total + right_total
        
        rows.append([left_idx, right_idx, merge_distance, left_size + right_size])
        node._linkage_id = next_id
        node._subtree_height = max(left_total, right_total)
        next_id += 1
        return node._linkage_id, node._subtree_height, left_size + right_size
    
    postorder(root)
    return np.asarray(rows, dtype=float)


def linkage_to_newick(Z, labels, support_dict=None, out_path=None):
    """Convert linkage matrix to Newick string."""
    tree = hierarchy.to_tree(Z, rd=False)
    
    def build(node, parent_dist):
        node_dist = node.dist if node.dist is not None else 0.0
        branch_length = max(parent_dist - node_dist, 0.0)
        
        if node.left is None and node.right is None:
            # Leaf
            name = labels[node.id] if node.id < len(labels) else f'leaf{node.id}'
            return f'{name}:{branch_length:.10f}', {name}
        
        # Internal node
        left_str, left_clade = build(node.left, node_dist)
        right_str, right_clade = build(node.right, node_dist)
        clade = left_clade | right_clade
        
        support_label = ''
        if support_dict:
            support = support_dict.get(frozenset(clade))
            if support is not None:
                support_label = f'{support:.1f}'
        
        inner = f'({left_str},{right_str}){support_label}'
        
        if parent_dist == node_dist:
            return inner, clade
        return f'{inner}:{branch_length:.10f}', clade
    
    root_dist = tree.dist if tree.dist is not None else 0.0
    newick, _ = build(tree, root_dist)
    
    if not newick.endswith(';'):
        newick += ';'
    
    if out_path:
        with open(out_path, 'w') as fh:
            fh.write(newick)
    
    return newick


def tree_to_newick_string(tree):
    """Convert skbio tree to Newick string."""
    buf = io.StringIO()
    tree.write(buf)
    return buf.getvalue().strip()


def write_pseudotrees(newicks, out_path):
    """Write list of Newick strings to file."""
    if not newicks:
        return
    with open(out_path, 'w') as fh:
        for nw in newicks:
            line = nw.strip()
            if not line.endswith(';'):
                line += ';'
            fh.write(line + '\n')


# ==============================================================================
# Bootstrap
# ==============================================================================

def ensure_nonzero_rows(X, eps=1e-8):
    """Ensure no all-zero rows in matrix."""
    norms = np.linalg.norm(X, axis=1)
    zero_mask = norms < eps
    if zero_mask.any():
        X = X.copy()
        X[zero_mask, 0] = eps
    return X


def collect_linkage_clades(node, ids):
    """Collect all clades from linkage tree."""
    clades = []
    
    def helper(n):
        if n.left is None and n.right is None:
            return {ids[n.id]}
        left = helper(n.left)
        right = helper(n.right)
        leaves = left | right
        if 1 < len(leaves) < len(ids):
            clades.append((n, frozenset(leaves)))
        return leaves
    
    helper(node)
    return clades


def get_internal_clades(tree):
    """Get all internal clades from tree."""
    clades = []
    n_tips = len(list(tree.tips()))
    
    for node in tree.non_tips():
        node_tips = frozenset(clean_tip_name(tip.name) for tip in node.tips())
        if 1 < len(node_tips) < n_tips:
            clades.append((node, node_tips))
    
    return clades


def _upgma_bootstrap_worker(seed, X, ids, subset_size):
    """Bootstrap worker for UPGMA."""
    rng = np.random.default_rng(seed)
    cols = rng.choice(X.shape[1], size=subset_size, replace=False)
    X_sub = ensure_nonzero_rows(X[:, cols])
    
    # Compute cosine distances
    dist_sub = compute_cosine_distances(X_sub)
    condensed_sub = squareform(dist_sub)
    
    # Build tree
    Z_sub = hierarchy.linkage(condensed_sub, method='average')
    boot_tree = hierarchy.to_tree(Z_sub, rd=False)
    boot_clades = {clade for _, clade in collect_linkage_clades(boot_tree, ids)}
    
    return boot_clades, linkage_to_newick(Z_sub, ids)


def bootstrap_support_upgma(X, ids, base_Z,
                            n_reps=200, subsample_ratio=0.8,
                            random_state=0, n_jobs=1):
    """Compute bootstrap support for UPGMA tree."""
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    subset_size = max(2, int(round(n_features * subsample_ratio)))
    
    base_tree = hierarchy.to_tree(base_Z, rd=False)
    ref_clades = collect_linkage_clades(base_tree, ids)
    support_counts = {clade: 0 for _, clade in ref_clades}
    
    seeds = [int(s) for s in rng.integers(0, 2**32 - 1, size=n_reps)]
    
    worker = lambda seed: _upgma_bootstrap_worker(seed, X, ids, subset_size)
    
    if n_jobs == 1:
        results = [worker(s) for s in seeds]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(worker, seeds))
    
    pseudo_newicks = []
    for boot_clades, newick in results:
        pseudo_newicks.append(newick)
        for _, clade in ref_clades:
            if clade in boot_clades:
                support_counts[clade] += 1
    
    support_dict = {clade: 100.0 * count / n_reps
                   for clade, count in support_counts.items()}
    
    return support_dict, pseudo_newicks


def _nj_bootstrap_worker(seed, X, ids, subset_size):
    """Bootstrap worker for NJ."""
    rng = np.random.default_rng(seed)
    cols = rng.choice(X.shape[1], size=subset_size, replace=False)
    X_sub = ensure_nonzero_rows(X[:, cols])
    
    # Compute cosine distances
    dist_sub = compute_cosine_distances(X_sub)
    dist_sub = validate_distance_matrix(dist_sub)
    
    # Build tree
    dm = DistanceMatrix(dist_sub, ids)
    boot_tree = midpoint_root(nj(dm))
    
    for tip in boot_tree.tips():
        tip.name = clean_tip_name(tip.name)
    
    boot_clades = {frozenset(clean_tip_name(tip.name) for tip in node.tips())
                  for node in boot_tree.non_tips()
                  if 1 < len(list(node.tips())) < len(ids)}
    
    return boot_clades, tree_to_newick_string(boot_tree)


def bootstrap_support_nj(X, ids, base_tree,
                         n_reps=200, subsample_ratio=0.8,
                         random_state=0, n_jobs=1):
    """Compute bootstrap support for NJ tree."""
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    subset_size = max(2, int(round(n_features * subsample_ratio)))
    
    ref_clades = get_internal_clades(base_tree)
    support_counts = {clade: 0 for _, clade in ref_clades}
    
    seeds = [int(s) for s in rng.integers(0, 2**32 - 1, size=n_reps)]
    
    worker = lambda seed: _nj_bootstrap_worker(seed, X, ids, subset_size)
    
    if n_jobs == 1:
        results = [worker(s) for s in seeds]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(worker, seeds))
    
    pseudo_newicks = []
    for boot_clades, newick in results:
        pseudo_newicks.append(newick)
        for _, clade in ref_clades:
            if clade in boot_clades:
                support_counts[clade] += 1
    
    support_dict = {clade: 100.0 * count / n_reps
                   for clade, count in support_counts.items()}
    
    tree_with_support = base_tree.copy()
    for node, clade in get_internal_clades(tree_with_support):
        support = support_dict.get(clade, 0.0)
        node.name = f'{support:.1f}'
    
    return tree_with_support, support_dict, pseudo_newicks


# ==============================================================================
# Partition Quality Metrics
# ==============================================================================
def compute_wss(X, labels):
    """Compute Within-cluster Sum of Squares."""
    wss = 0.0
    for label in np.unique(labels):
        mask = labels == label
        cluster_data = X[mask]
        if len(cluster_data) > 0:
            centroid = cluster_data.mean(axis=0)
            wss += np.sum((cluster_data - centroid) ** 2)
    return wss


def compute_bcubed_metrics(true_labels, pred_labels):
    """
    Compute BCubed Precision, Recall, and F-score using vectorized operations.
    """
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)
    if not isinstance(pred_labels, np.ndarray):
        pred_labels = np.array(pred_labels)
    
    n = len(true_labels)
    true_pairs = true_labels[:, None] == true_labels[None, :]
    pred_pairs = pred_labels[:, None] == pred_labels[None, :]
    intersection = true_pairs & pred_pairs
    
    # Precision: for each sample, intersection / predicted_cluster_size
    pred_cluster_sizes = pred_pairs.sum(axis=1)  # size of predicted cluster for each sample
    precision_per_sample = intersection.sum(axis=1) / pred_cluster_sizes
    precision = precision_per_sample.mean()
    
    # Recall: for each sample, intersection / true_cluster_size
    true_cluster_sizes = true_pairs.sum(axis=1)  # size of true cluster for each sample
    recall_per_sample = intersection.sum(axis=1) / true_cluster_sizes
    recall = recall_per_sample.mean()
    
    # F-score
    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0
    
    return precision, recall, fscore


def compute_splitting_lumping_indices(true_labels, pred_labels):
    """
    Compute splitting and lumping indices.
    
    Returns:
        splitting_index: avg number of OTUs per true species
        lumping_index: avg number of true species per OTU
    """
    species_to_otus = defaultdict(set)
    otu_to_species = defaultdict(set)
    
    for t, p in zip(true_labels, pred_labels):
        species_to_otus[t].add(p)
        otu_to_species[p].add(t)
    
    splitting_counts = [len(otus) for otus in species_to_otus.values()]
    splitting_index = np.mean(splitting_counts)
    
    lumping_counts = [len(species) for species in otu_to_species.values()]
    lumping_index = np.mean(lumping_counts)
    
    return splitting_index, lumping_index

def compute_cluster_purity(true_labels, pred_labels):
    """
    Compute clustering purity.
    
    Purity measures the extent to which each cluster contains samples 
    from primarily one class. For each cluster, we find the most common 
    true label and compute the fraction of correctly assigned samples.
    
    Args:
        true_labels: Ground truth labels [N]
        pred_labels: Predicted cluster labels [N]
    
    Returns:
        purity: Purity score in [0, 1], higher is better
    """
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)
    if not isinstance(pred_labels, np.ndarray):
        pred_labels = np.array(pred_labels)
    
    # Build contingency matrix: rows=clusters, cols=true_classes
    unique_clusters = np.unique(pred_labels)
    unique_classes = np.unique(true_labels)
    
    n_clusters = len(unique_clusters)
    n_classes = len(unique_classes)
    
    # Create mapping to indices
    cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    
    contingency = np.zeros((n_clusters, n_classes), dtype=int)
    
    for true_label, pred_label in zip(true_labels, pred_labels):
        cluster_idx = cluster_to_idx[pred_label]
        class_idx = class_to_idx[true_label]
        contingency[cluster_idx, class_idx] += 1
    
    # Purity = sum of max counts per cluster / total samples
    # Each cluster contributes its most frequent class
    purity = np.sum(np.max(contingency, axis=1)) / len(true_labels)
    
    return purity


def compute_monophyly_proportion(tree, true_labels, ids_tree):
    """Compute proportion of monophyletic species in tree."""
    try:
        from skbio import TreeNode
        is_skbio = isinstance(tree, TreeNode)
    except:
        is_skbio = False
    
    id_to_label = {ids_tree[i]: true_labels[i] for i in range(len(ids_tree))}
    unique_species = set(true_labels)
    
    monophyletic = set()
    non_monophyletic = set()
    
    if is_skbio:
        for species in unique_species:
            species_tips = {tip_id for tip_id, label in id_to_label.items() 
                          if label == species}
            
            tips_to_check = [tip for tip in tree.tips() 
                           if clean_tip_name(tip.name) in species_tips]
            
            if len(tips_to_check) == 1:
                monophyletic.add(species)
                continue
            
            lca = tips_to_check[0].lowest_common_ancestor(tips_to_check[1:])
            lca_tips = {clean_tip_name(tip.name) for tip in lca.tips()}
            lca_species = {id_to_label.get(tip_id) for tip_id in lca_tips}
            
            if lca_species == {species}:
                monophyletic.add(species)
            else:
                non_monophyletic.add(species)
    
    else:
        print("  [WARNING] Monophyly check for linkage matrix is approximate")
        return None, None, None
    
    total_species = len(unique_species)
    if total_species > 0:
        proportion = len(monophyletic) / total_species
    else:
        proportion = 0.0
    
    return proportion, list(monophyletic), list(non_monophyletic)


def compute_silhouette_score_safe(X, labels, metric='cosine'):
    """Compute Silhouette Score safely (handles edge cases)."""
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import normalize
    except ImportError:
        return 0.0
    
    if len(X) < 2:
        return 0.0
    
    if metric == 'cosine':
        X = normalize(X, norm='l2')
    
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0
    
    try:
        return silhouette_score(X, labels, metric=metric)
    except Exception as e:
        print(f"  [WARNING] Silhouette score computation failed: {e}")
        return 0.0


def assess_partition_quality(X_tree, ids_tree, true_labels, partitions, 
                             tree=None, n_jobs=1, sample_size=10000):
    """
    Assess partition quality using multiple metrics.
    
    Metrics are divided into categories:
    1. Tree-level (full dataset, computed once): monophyly_proportion
    2. Species-level (sampled data, computed once): silhouette_species
    3. Partition-based (full dataset, per cutoff): n_OTUs, splitting/lumping indices
    4. Sample-based (sampled data, per cutoff): clustering metrics, WSS
    5. Cluster-quality (full dataset, per cutoff): silhouette_cluster (cluster-stratified)
    
    Args:
        X_tree: Embedding data (n, d)
        ids_tree: Sample IDs
        true_labels: Ground truth labels (or None for unlabeled metrics only)
        partitions: Dict of {cutoff: cluster_labels}
        tree: Tree object (optional, for monophyly)
        n_jobs: Number of parallel jobs
        sample_size: Number of samples for sample-based metrics (0=use all)
    
    Returns:
        metrics_df: DataFrame with metrics for each cutoff
    """
    from sklearn.metrics import (
        adjusted_rand_score, 
        normalized_mutual_info_score,
        adjusted_mutual_info_score,
        homogeneity_completeness_v_measure
    )
    
    print(f"\n  Partition quality assessment:")
    
    # Check label validity
    has_labels = true_labels is not None and len([x for x in true_labels if x is not None]) > 0
    n_samples = len(ids_tree)
    
    # Filter valid samples (with labels)
    if has_labels:
        valid_mask = np.array([x is not None for x in true_labels])
        X_valid = X_tree[valid_mask]
        true_labels_valid = true_labels[valid_mask]
        ids_valid = [ids_tree[i] for i in range(len(ids_tree)) if valid_mask[i]]
        n_true_species = len(np.unique(true_labels_valid))
        n_valid = len(X_valid)
        
        print(f"    Dataset: {n_valid} samples, {n_true_species} true species")
    else:
        valid_mask = np.ones(n_samples, dtype=bool)
        X_valid = X_tree
        true_labels_valid = None
        ids_valid = ids_tree
        n_valid = n_samples
        
        print(f"    Dataset: {n_valid} samples (unlabeled)")
    
    # === STAGE 1: Tree-level metrics (computed once, full dataset) ===
    print(f"\n    [Stage 1/3] Computing tree-level metrics...")
    
    monophyly_prop = None
    
    if has_labels and tree is not None:
        print(f"      - Monophyly proportion (full dataset)", end='', flush=True)
        try:
            # Use ORIGINAL full dataset for monophyly
            monophyly_prop, _, _ = compute_monophyly_proportion(
                tree, true_labels, ids_tree
            )
            print(f" = {monophyly_prop:.4f}")
        except Exception as e:
            print(f" [FAILED: {e}]")
    
    # === STAGE 2: Determine sampling strategy ===
    use_sampling = (sample_size > 0) and (n_valid > sample_size)
    
    if use_sampling:
        print(f"\n    [Stage 2/3] Sampling for intensive metrics...")
        print(f"      Target: {sample_size} samples from {n_valid}")
        
        rng = np.random.default_rng(42)
        
        if has_labels:
            # Stratified sampling by label
            from collections import defaultdict
            label_indices = defaultdict(list)
            for i, label in enumerate(true_labels_valid):
                label_indices[label].append(i)
            
            sample_indices = []
            for label, indices in label_indices.items():
                n_class = len(indices)
                n_sample_class = max(1, int(sample_size * n_class / n_valid))
                n_sample_class = min(n_sample_class, n_class)
                sampled = rng.choice(indices, size=n_sample_class, replace=False)
                sample_indices.extend(sampled)
            
            sample_indices = np.array(sample_indices)
            if len(sample_indices) > sample_size:
                sample_indices = rng.choice(sample_indices, size=sample_size, replace=False)
            
            print(f"      Strategy: Stratified by species")
        else:
            # Random sampling
            sample_indices = rng.choice(n_valid, size=sample_size, replace=False)
            print(f"      Strategy: Random")
        
        X_sample = X_valid[sample_indices]
        true_labels_sample = true_labels_valid[sample_indices] if has_labels else None
        
        # Create sample mask in valid space
        sample_mask_valid = np.zeros(n_valid, dtype=bool)
        sample_mask_valid[sample_indices] = True
        
        print(f"      Result: {len(X_sample)} samples selected")
        
    else:
        print(f"\n    [Stage 2/3] Using all samples (no sampling)")
        X_sample = X_valid
        true_labels_sample = true_labels_valid
        sample_mask_valid = np.ones(n_valid, dtype=bool)
    
    # === Compute silhouette_species once on sampled data ===
    silhouette_species_score = None
    if has_labels:
        print(f"      - Silhouette score (species-level, {len(X_sample)} samples)", end='', flush=True)
        silhouette_species_score = compute_silhouette_score_safe(
            X_sample, true_labels_sample, metric='cosine'
        )
        print(f" = {silhouette_species_score:.4f}")
    
    # === Helper function for approximate silhouette_cluster ===
    def compute_silhouette_cluster_approximate(X_full, labels_full, max_per_cluster=100):
        """
        Compute approximate silhouette score using cluster-stratified sampling.
        
        This preserves cluster structure better than random global sampling
        and is much faster for large datasets.
        
        Args:
            X_full: Full dataset embeddings
            labels_full: Cluster labels for full dataset
            max_per_cluster: Maximum samples per cluster (default: 100)
        
        Returns:
            Approximate silhouette score
        """
        unique_labels = np.unique(labels_full)
        n_clusters = len(unique_labels)
        
        # Use full data if small dataset or few clusters
        if len(X_full) <= 5000 or n_clusters <= 5:
            return compute_silhouette_score_safe(X_full, labels_full, metric='cosine')
        
        # Cluster-stratified sampling
        rng = np.random.default_rng(42)
        sample_indices = []
        
        for label in unique_labels:
            cluster_indices = np.where(labels_full == label)[0]
            n_cluster = len(cluster_indices)
            
            if n_cluster <= max_per_cluster:
                # Use all samples from small clusters
                sample_indices.extend(cluster_indices)
            else:
                # Sample from large clusters
                sampled = rng.choice(cluster_indices, size=max_per_cluster, replace=False)
                sample_indices.extend(sampled)
        
        sample_indices = np.array(sample_indices)
        X_sampled = X_full[sample_indices]
        labels_sampled = labels_full[sample_indices]
        
        return compute_silhouette_score_safe(X_sampled, labels_sampled, metric='cosine')
    
    # === STAGE 3: Per-cutoff metrics ===
    print(f"\n    [Stage 3/3] Computing per-cutoff metrics...")
    print(f"      Cutoffs to process: {len(partitions)}")
    
    if has_labels:
        print(f"\n      Partition-based (full dataset):")
        print(f"        - n_OTUs, OTU_species_ratio")
        print(f"        - splitting_index, lumping_index")
        print(f"\n      Sample-based ({len(X_sample)} samples):")
        print(f"        - ARI, NMI, AMI")
        print(f"        - homogeneity, completeness, v_measure")
        print(f"        - purity")
        print(f"        - BCubed precision/recall/F-score")
        print(f"        - WSS")
        print(f"\n      Cluster-quality (full dataset, approximate):")
        print(f"        - silhouette_cluster (max {100} samples per cluster)")
    else:
        print(f"\n      Metrics:")
        print(f"        - n_OTUs ({n_valid} samples)")
        print(f"        - silhouette_cluster (full dataset, approximate)")
        print(f"        - WSS ({len(X_sample)} samples)")
    
    def compute_metrics_for_cutoff(item):
        cutoff, pred_labels_full = item
        
        metrics = {'cutoff': cutoff}
        
        # Extract labels for valid and sampled subsets
        pred_labels_valid = pred_labels_full[valid_mask]
        pred_labels_sample = pred_labels_valid[sample_mask_valid]
        
        if has_labels:
            # === Partition-based metrics (FULL dataset) ===
            n_otus_full = len(np.unique(pred_labels_valid))
            metrics['n_OTUs'] = n_otus_full
            metrics['n_true_species'] = n_true_species
            metrics['OTU_species_ratio'] = n_otus_full / n_true_species if n_true_species > 0 else 0.0
            
            # Splitting/Lumping indices (FULL dataset)
            try:
                split_idx, lump_idx = compute_splitting_lumping_indices(
                    true_labels_valid, pred_labels_valid
                )
                metrics['splitting_index'] = split_idx
                metrics['lumping_index'] = lump_idx
            except:
                metrics['splitting_index'] = 0.0
                metrics['lumping_index'] = 0.0
            
            # === Sample-based metrics ===
            # Clustering agreement
            metrics['ARI'] = adjusted_rand_score(true_labels_sample, pred_labels_sample)
            metrics['NMI'] = normalized_mutual_info_score(true_labels_sample, pred_labels_sample)
            metrics['AMI'] = adjusted_mutual_info_score(true_labels_sample, pred_labels_sample)
            
            # Homogeneity
            h, c, v = homogeneity_completeness_v_measure(true_labels_sample, pred_labels_sample)
            metrics['homogeneity'] = h
            metrics['completeness'] = c
            metrics['v_measure'] = v
            
            # Purity
            try:
                purity = compute_cluster_purity(true_labels_sample, pred_labels_sample)
                metrics['purity'] = purity
            except:
                metrics['purity'] = 0.0
            
            # BCubed
            try:
                bcubed_p, bcubed_r, bcubed_f = compute_bcubed_metrics(
                    true_labels_sample, pred_labels_sample
                )
                metrics['BCubed_precision'] = bcubed_p
                metrics['BCubed_recall'] = bcubed_r
                metrics['BCubed_fscore'] = bcubed_f
            except:
                metrics['BCubed_precision'] = 0.0
                metrics['BCubed_recall'] = 0.0
                metrics['BCubed_fscore'] = 0.0
            
            # WSS (on sampled data)
            metrics['WSS'] = compute_wss(X_sample, pred_labels_sample)
            
            # === Cluster-quality metrics (FULL dataset with approximate silhouette) ===
            metrics['silhouette_cluster'] = compute_silhouette_cluster_approximate(
                X_valid, pred_labels_valid, max_per_cluster=1000
            )
            
            # === Global metrics (same for all cutoffs) ===
            if monophyly_prop is not None:
                metrics['monophyly_proportion'] = monophyly_prop
            
            if silhouette_species_score is not None:
                metrics['silhouette_species'] = silhouette_species_score
        
        else:
            # Unlabeled case
            n_otus = len(np.unique(pred_labels_valid))
            metrics['n_OTUs'] = n_otus
            
            # Silhouette (approximate, on full dataset)
            metrics['silhouette_cluster'] = compute_silhouette_cluster_approximate(
                X_valid, pred_labels_valid, max_per_cluster=1000
            )
            
            # WSS (on sampled data)
            metrics['WSS'] = compute_wss(X_sample, pred_labels_sample)
        
        return metrics
    
    # Compute metrics (parallel or serial)
    if n_jobs == 1:
        results = [compute_metrics_for_cutoff(item) for item in partitions.items()]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(compute_metrics_for_cutoff, partitions.items()))
    
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df.sort_values('cutoff').reset_index(drop=True)
    
    print(f"\n    ✓ Metrics computation complete")
    
    return metrics_df



def plot_metrics_dashboard(metrics_df, out_path, has_labels=True):
    """Plot comprehensive metrics dashboard."""
    if has_labels:
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        
        # Row 1
        axes[0, 0].plot(metrics_df['cutoff'], metrics_df['n_OTUs'], 'o-', color='#1f77b4')
        axes[0, 0].axhline(y=metrics_df['n_true_species'].iloc[0], 
                          color='red', linestyle='--', label='True species')
        axes[0, 0].set_xlabel('Distance cutoff')
        axes[0, 0].set_ylabel('Number of OTUs')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(metrics_df['cutoff'], metrics_df['ARI'], 'o-', color='#2ca02c', label='ARI')
        axes[0, 1].plot(metrics_df['cutoff'], metrics_df['NMI'], 's-', color='#ff7f0e', label='NMI')
        axes[0, 1].plot(metrics_df['cutoff'], metrics_df['AMI'], '^-', color='#d62728', label='AMI')
        axes[0, 1].set_xlabel('Distance cutoff')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Clustering Agreement')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(metrics_df['cutoff'], metrics_df['BCubed_fscore'], 'o-', color='#9467bd')
        axes[0, 2].set_xlabel('Distance cutoff')
        axes[0, 2].set_ylabel('BCubed F-score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2
        axes[1, 0].plot(metrics_df['cutoff'], metrics_df['splitting_index'], 'o-', color='#8c564b')
        axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Distance cutoff')
        axes[1, 0].set_ylabel('Splitting index')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(metrics_df['cutoff'], metrics_df['lumping_index'], 'o-', color='#e377c2')
        axes[1, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Distance cutoff')
        axes[1, 1].set_ylabel('Lumping index')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(metrics_df['cutoff'], metrics_df['silhouette_species'], 
                       'o-', color='#7f7f7f', label='Species-level')
        axes[1, 2].plot(metrics_df['cutoff'], metrics_df['silhouette_cluster'], 
                       's-', color='#bcbd22', label='Cluster-level')
        axes[1, 2].set_xlabel('Distance cutoff')
        axes[1, 2].set_ylabel('Silhouette score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Row 3
        axes[2, 0].plot(metrics_df['cutoff'], metrics_df['homogeneity'], 
                       'o-', color='#17becf', label='Homogeneity')
        axes[2, 0].plot(metrics_df['cutoff'], metrics_df['completeness'], 
                       's-', color='#ff9896', label='Completeness')
        axes[2, 0].plot(metrics_df['cutoff'], metrics_df['v_measure'], 
                       '^-', color='#98df8a', label='V-measure')
        axes[2, 0].set_xlabel('Distance cutoff')
        axes[2, 0].set_ylabel('Score')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(metrics_df['n_OTUs'], metrics_df['WSS'], 'o-', color='#c5b0d5')
        axes[2, 1].set_xlabel('Number of OTUs')
        axes[2, 1].set_ylabel('WSS')
        axes[2, 1].set_title('WSS Elbow')
        axes[2, 1].grid(True, alpha=0.3)
        
        axes[2, 2].plot(metrics_df['cutoff'], metrics_df['purity'], 
                       'o-', color='#1f77b4', linewidth=2)
        axes[2, 2].set_xlabel('Distance cutoff')
        axes[2, 2].set_ylabel('Cluster Purity')
        axes[2, 2].set_title('Cluster Purity')
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_ylim([0, 1.05])  # Purity is in [0, 1]
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(metrics_df['cutoff'], metrics_df['n_OTUs'], 'o-', color='#1f77b4')
        axes[0].set_xlabel('Distance cutoff')
        axes[0].set_ylabel('Number of OTUs')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(metrics_df['cutoff'], metrics_df['silhouette_cluster'], 
                    'o-', color='#2ca02c', linewidth=2)
        axes[1].set_xlabel('Distance cutoff')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Cluster Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(metrics_df['n_OTUs'], metrics_df['WSS'], 'o-', color='#c5b0d5')
        axes[2].set_xlabel('Number of OTUs')
        axes[2].set_ylabel('WSS')
        axes[2].set_title('WSS Elbow')
        axes[2].grid(True, alpha=0.3)
    
    fig.suptitle('Partition Quality Metrics Dashboard', fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved metrics dashboard to {out_path}")


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_pipeline(args):
    """Main analysis pipeline."""
    # Set CPU thread limits for numerical libraries
    os.environ["OMP_NUM_THREADS"] = str(args.cpus)
    os.environ["MKL_NUM_THREADS"] = str(args.cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.cpus)
    
    # Set PyTorch threads if available
    try:
        import torch
        torch.set_num_threads(args.cpus)
    except ImportError:
        pass

    print("\n" + "=" * 80)
    print("PCA Whitening → Cosine Distance → Local Scaling → Tree Construction")
    print("=" * 80)
    
    # ===== Load Data =====
    print("\n[1/7] Loading embeddings...")
    
    if args.tree_data == "test_only":
        df_train = None
        X_train = None
        ids_train = None
        
        df_test = load_embeddings(args.embeddings_test_csv)
        X_test, ids_test = prepare_embeddings(df_test)
        print(f"  Test:  {X_test.shape[0]} samples × {X_test.shape[1]} features")
        print(f"  (test_only mode: no training data loaded)")
    
    else:
        df_train = load_embeddings(args.embeddings_train_csv)
        X_train, ids_train = prepare_embeddings(df_train)
        print(f"  Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
        
        df_test = None
        X_test = None
        ids_test = None
        
        if args.embeddings_test_csv:
            df_test = load_embeddings(args.embeddings_test_csv)
            X_test, ids_test = prepare_embeddings(df_test)
            print(f"  Test:  {X_test.shape[0]} samples × {X_test.shape[1]} features")
    

    # ===== PCA Whitening =====
    if args.pca_whitening:
        print("\n[2/7] Applying PCA whitening...")
        
        if args.tree_data == "test_only":
            # test_only
            print("  (test_only mode: fitting PCA on test data)")
            pca = PCA(n_components=args.pca_components, whiten=True, 
                     random_state=args.random_state)
            X_test = pca.fit_transform(X_test)
            explained_var = pca.explained_variance_ratio_.sum()
            print(f"  PCA explained variance: {explained_var:.4f}")
        
        else:
            # other modes
            X_train, X_test, pca = apply_pca_whitening(
                X_train, X_test,
                n_components=args.pca_components,
                random_state=args.random_state
            )
    else:
        print("\n[2/7] Skipping PCA whitening (using raw embeddings)")
    
    # ===== Select Data for Tree =====
    print(f"\n[3/7] Selecting data for tree construction (mode: {args.tree_data})...")
    
    if args.tree_data == "train":
        X_tree = X_train
        ids_tree = ids_train
        print(f"  Using train data only: {X_tree.shape[0]} samples")
    
    elif args.tree_data == "test":
        if X_test is None:
            raise ValueError("--tree_data=test requires --embeddings_test_csv")
        X_tree = X_test
        ids_tree = ids_test
        print(f"  Using test data only (with train-fitted PCA): {X_tree.shape[0]} samples")
    
    elif args.tree_data == "test_only":
        X_tree = X_test
        ids_tree = ids_test
        print(f"  Using test data only (independent mode): {X_tree.shape[0]} samples")
    
    elif args.tree_data == "train_test":
        if X_test is None:
            raise ValueError("--tree_data=train_test requires --embeddings_test_csv")
        X_tree = np.vstack([X_train, X_test])
        ids_tree = ids_train + ids_test
        print(f"  Using combined train+test: {X_tree.shape[0]} samples")

    
    # Fix zero vectors
    X_tree, ids_tree = remove_or_fix_zero_vectors(X_tree, ids_tree)

    # ===== Load Labels =====
    true_labels = None
    id_to_label = None
    if args.label_data_csv:
        print("\n  Loading label data...")
        true_labels, id_to_label = load_label_data(args.label_data_csv, ids_tree)
    
    # ===== Compute Distances =====
    print("\n[4/7] Computing cosine distances...")
    dist_raw = compute_cosine_distances(X_tree)
    dist_raw = validate_distance_matrix(dist_raw)
    
    stats_raw = compute_distance_stats(dist_raw, name="raw_cosine")
    print(f"  Raw cosine distance range: [{stats_raw['min']:.4f}, {stats_raw['max']:.4f}]")
    print(f"  Mean: {stats_raw['mean']:.4f}, Median: {stats_raw['median']:.4f}")

    #print("  Checking memory usage and releasing unnecessary data...")
    need_X_tree = (args.num_bootstraps > 0) or args.assess_partition_metrics
    
    if not need_X_tree:
        del X_tree
        if 'X_train' in locals() and X_train is not None:
            del X_train
        if 'X_test' in locals() and X_test is not None:
            del X_test
        gc.collect()
        #print(f"    Released embedding data (not needed for remaining steps)")
    else:
        if 'X_train' in locals() and X_train is not None:
            del X_train
        if 'X_test' in locals() and X_test is not None:
            del X_test
        if 'df_train' in locals():
            del df_train
        if 'df_test' in locals():
            del df_test
        gc.collect()
        #print(f"    Released intermediate data, kept X_tree for bootstrap/metrics")

    # ===== Local Scaling =====
    if not args.enable_local_scaling:
        print("\n[5/7] Local scaling disabled (using raw cosine distances)")
        dist_final = dist_raw
        stats_final = stats_raw
        distance_mode = "Cosine"
    else:
        print("\n[5/7] Applying local scaling (Mutual Proximity)...")
        
        if args.local_k > 0:
            k = args.local_k
            print(f"  Using manual k={k}")
        else:
            k = auto_select_k(X_tree.shape[0], strategy=args.local_k_strategy)
            print(f"  Auto-selected k={k} (strategy={args.local_k_strategy})")
        
        dist_scaled, knn_dists = apply_local_scaling(dist_raw, k)
        
        stats_scaled = compute_distance_stats(dist_scaled, name="local_scaled")
        print(f"  Scaled distance range: [{stats_scaled['min']:.4f}, {stats_scaled['max']:.4f}]")
        print(f"  Mean: {stats_scaled['mean']:.4f}, Median: {stats_scaled['median']:.4f}")
        
        dist_final = dist_scaled
        stats_final = stats_scaled
        distance_mode = "CosineScaled"
    
    # ===== Save Statistics =====
    base_dir = ensure_dir(args.out_dir)
    stats_dir = ensure_dir(os.path.join(base_dir, "distance_statistics"))
    
    all_stats = [stats_raw]
    if args.enable_local_scaling:
        all_stats.append(stats_final)
    
    save_distance_statistics(all_stats, os.path.join(stats_dir, "distance_stats.json"))
    
    # ===== Plot Distributions =====
    print("\n[6/7] Plotting distance distributions...")
    
    plot_distance_distributions(
        dist_raw, "raw_cosine", stats_dir,
        metric="cosine", max_pairs=args.max_distance_pairs,
        random_state=args.random_state
    )
    
    if args.enable_local_scaling:
        plot_distance_distributions(
            dist_final, "local_scaled", stats_dir,
            metric="cosine", max_pairs=args.max_distance_pairs,
            random_state=args.random_state
        )
    
    # ===== Save Distance Matrices =====
    if args.save_pairwise_distances:
        print("\n  Saving pairwise distance matrices...")
        save_distance_matrix(
            dist_raw, ids_tree,
            os.path.join(stats_dir, "pairwise_distances_raw.csv")
        )
        if args.enable_local_scaling:
            save_distance_matrix(
                dist_final, ids_tree,
                os.path.join(stats_dir, "pairwise_distances_scaled.csv")
            )

    if args.label_data_csv and id_to_label is not None:
        print("\n  Computing intra-class distance statistics...")
        
        # Compute for raw distances
        compute_intraclass_distance_statistics(
            dist_raw, ids_tree, id_to_label,
            os.path.join(stats_dir, "pairwise_distance_summary_intra-class_raw.csv")
        )
        
        # Compute for scaled distances if applicable
        if args.enable_local_scaling:
            compute_intraclass_distance_statistics(
                dist_final, ids_tree, id_to_label,
                os.path.join(stats_dir, "pairwise_distance_summary_intra-class_scaled.csv")
            )
    
    # ===== Determine Cutoffs =====
    print("\n[7/7] Determining cutoffs and building trees...")
    
    cutoffs = determine_cutoffs(args)
    
    # ===== Thread Pool =====
    n_jobs = args.cpus
    
    # ===== UPGMA Workflow =====
    if args.UPGMA:
        print("\n" + "=" * 80)
        print("UPGMA Tree Construction")
        print("=" * 80)
        
        upgma_dir = ensure_dir(os.path.join(base_dir, "UPGMA"))
        
        # Build tree
        print("  Building UPGMA tree...")
        condensed = squareform(dist_final)
        Z_upgma = hierarchy.linkage(condensed, method="average")
        
        # Save tree
        linkage_to_newick(
            Z_upgma, ids_tree,
            out_path=os.path.join(upgma_dir, f"UPGMA_{distance_mode}.nwk")
        )
        
        # Bootstrap
        support_upgma = {}
        if args.num_bootstraps > 0:
            print(f"  Running {args.num_bootstraps} bootstrap replicates...")
            support_upgma, pseudo_upgma = bootstrap_support_upgma(
                X_tree, ids_tree, Z_upgma,
                n_reps=args.num_bootstraps,
                subsample_ratio=args.bootstrap_subsample_ratio,
                random_state=args.random_state,
                n_jobs=args.cpus
            )
            
            linkage_to_newick(
                Z_upgma, ids_tree, support_dict=support_upgma,
                out_path=os.path.join(upgma_dir, f"UPGMA_{distance_mode}_bootstrap.nwk")
            )
            
            if args.save_bootstrap_trees:
                write_pseudotrees(
                    pseudo_upgma,
                    os.path.join(upgma_dir, f"UPGMA_{distance_mode}_bootstrap.trees")
                )
            
            print("  Bootstrap complete.")
        
        # Partitions
        print("  Computing partitions...")
        partitions_dir = ensure_dir(os.path.join(upgma_dir, "partitions"))
        
        scan_linkage_thresholds(
            Z_upgma, cutoffs,
            label=f"UPGMA ({distance_mode})",
            out_csv=os.path.join(partitions_dir, "partition_scan.csv"),
            out_fig=os.path.join(partitions_dir, "partition_scan.pdf")
        )
        
        partitions_upgma = build_partitions_from_linkage(Z_upgma, cutoffs)
        export_partition_tables(
            ids_tree, partitions_upgma,
            ensure_dir(os.path.join(partitions_dir, "tables"))
        )
        
        # Plot tree with partitions
        plot_upgma_partition_tree_panel(
            Z_upgma, ids_tree, partitions_upgma,
            out_path=os.path.join(partitions_dir, "UPGMA_tree_partitions.pdf"),
            tree_label=f"{distance_mode} distance",
            support_dict=support_upgma if args.num_bootstraps > 0 else None,
            bootstrap_cutoff=args.bootstrap_display_cutoff
        )

        if args.assess_partition_metrics:
            print("  Assessing partition quality metrics...")

            # Convert linkage to tree for monophyly calculation
            upgma_tree_for_metrics = None
            if true_labels is not None:
                try:
                    # Build tree from distance matrix for monophyly check
                    upgma_tree_for_metrics = build_nj_tree(
                        dist_final, ids_tree, midpoint=False, out_path=None
                    )
                except Exception as e:
                    print(f"  [WARNING] Could not build tree for monophyly: {e}")
            
            metrics_df = assess_partition_quality(
                X_tree, ids_tree, true_labels, partitions_upgma,
                tree=upgma_tree_for_metrics,
                n_jobs=n_jobs,
                sample_size=args.metrics_sample_size
            )
            
            metrics_csv = os.path.join(upgma_dir, "metrics.csv")
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"  Saved metrics to {metrics_csv}")
            
            has_labels = true_labels is not None
            plot_metrics_dashboard(
                metrics_df, 
                os.path.join(upgma_dir, "metrics_dashboard.pdf"),
                has_labels=has_labels
            )
            
        print(f"  UPGMA results saved to {upgma_dir}")
    
    # ===== NJ Workflow =====
    if args.NJ:
        print("\n" + "=" * 80)
        print("Neighbor-Joining Tree Construction")
        print("=" * 80)
        
        nj_dir = ensure_dir(os.path.join(base_dir, "NJ"))
        
        # Build tree
        print("  Building NJ tree...")
        nj_tree = build_nj_tree(
            dist_final, ids_tree, midpoint=True,
            out_path=os.path.join(nj_dir, f"NJ_{distance_mode}.nwk")
        )
        
        # Bootstrap
        support_nj = {}
        if args.num_bootstraps > 0:
            print(f"  Running {args.num_bootstraps} bootstrap replicates...")
            nj_tree_bootstrap, support_nj, pseudo_nj = bootstrap_support_nj(
                X_tree, ids_tree, base_tree=nj_tree,
                n_reps=args.num_bootstraps,
                subsample_ratio=args.bootstrap_subsample_ratio,
                random_state=args.random_state,
                n_jobs=args.cpus
            )
            
            nj_tree_bootstrap.write(
                os.path.join(nj_dir, f"NJ_{distance_mode}_bootstrap.nwk")
            )
            
            if args.save_bootstrap_trees:
                write_pseudotrees(
                    pseudo_nj,
                    os.path.join(nj_dir, f"NJ_{distance_mode}_bootstrap.trees")
                )
            
            print("  Bootstrap complete.")
        
        # Convert to linkage for partitioning
        Z_nj = tree_to_linkage(nj_tree, ids_tree)
        
        # Partitions
        print("  Computing partitions...")
        partitions_dir = ensure_dir(os.path.join(nj_dir, "partitions"))
        
        scan_linkage_thresholds(
            Z_nj, cutoffs,
            label=f"NJ ({distance_mode})",
            out_csv=os.path.join(partitions_dir, "partition_scan.csv"),
            out_fig=os.path.join(partitions_dir, "partition_scan.pdf")
        )
        
        partitions_nj = build_partitions_from_linkage(Z_nj, cutoffs)
        export_partition_tables(
            ids_tree, partitions_nj,
            ensure_dir(os.path.join(partitions_dir, "tables"))
        )
        
        # Plot tree with partitions
        plot_nj_partition_tree_panel(
            nj_tree, ids_tree, partitions_nj,
            out_path=os.path.join(partitions_dir, "NJ_tree_partitions.pdf"),
            tree_label=f"{distance_mode} distance",
            support_dict=support_nj if args.num_bootstraps > 0 else None,
            bootstrap_cutoff=args.bootstrap_display_cutoff
        )

        if args.assess_partition_metrics:
            print("  Assessing partition quality metrics...")
            
            metrics_df = assess_partition_quality(
                X_tree, ids_tree, true_labels, partitions_nj,
                tree=nj_tree,
                n_jobs=n_jobs,
                sample_size=args.metrics_sample_size
            )
            
            metrics_csv = os.path.join(nj_dir, "metrics.csv")
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"  Saved metrics to {metrics_csv}")
            
            has_labels = true_labels is not None
            plot_metrics_dashboard(
                metrics_df, 
                os.path.join(nj_dir, "metrics_dashboard.pdf"),
                has_labels=has_labels
            )
            
        print(f"  NJ results saved to {nj_dir}")
    
    # ===== Summary =====
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {Path(base_dir).resolve()}")
    print("\nDirectory structure:")
    print(f"  {base_dir}/")
    print(f"  ├── distance_statistics/")
    print(f"  │   ├── distance_stats.json")
    print(f"  │   ├── distance_hist_*.pdf")
    print(f"  │   └── distance_cum_*.pdf")
    if args.UPGMA:
        print(f"  ├── UPGMA/")
        print(f"  │   ├── UPGMA_{distance_mode}.nwk")
        print(f"  │   └── partitions/")
    if args.NJ:
        print(f"  └── NJ/")
        print(f"      ├── NJ_{distance_mode}.nwk")
        print(f"      └── partitions/")
    print("=" * 80 + "\n")


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    args = parse_args()

    # Log
    logs_dir = Path(args.out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "log.txt"
    
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger
    sys.stderr = tee_logger  # Also capture errors
    
    print(f"[Info] Logging to {log_file}")
    print(f"[Info] Command: {' '.join(sys.argv)}")
    #print(f"[Info] Arguments: {vars(args)}")

    run_pipeline(args)


if __name__ == "__main__":
    main()

