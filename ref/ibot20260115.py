#!/usr/bin/env python3
"""
SSL_domain_pretraining.py - Enhanced with comprehensive metrics
"""

import os
import sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import math
import random
import csv
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import timm
import torch
import shutil
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    from torchvision.transforms import TrivialAugmentWide
except ImportError:
    TrivialAugmentWide = None


# -------------------------------------------------------
# Enhanced Metrics Logger
# -------------------------------------------------------
class TeeLogger:
    """Redirect stdout to both console and file (skip progress bars)."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = log_path.open('a', encoding='utf-8')
        self.skip_patterns = [
            '\r',
            '|█', '|░',
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


class EnhancedMetricsLogger:
    """
    Comprehensive metrics logger for SSL and metric learning.
    Saves: NMI, ARI, Recall@K, kNN-Acc, Linear-Probing, mAP, Silhouette, Purity
    """
    def __init__(self, log_path: Path, mode: str = "pretrain"):
        self.log_path = log_path
        self.mode = mode
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Define fieldnames based on mode
        base_fields = ["epoch", "split"]
        metric_fields = [
            "NMI", "ARI",
            "Recall@1", "Recall@5", "Recall@10",
            "kNN_Acc_k1", "kNN_Acc_k5", "kNN_Acc_k20",
            "Linear_Probing_Acc",
            "mAP",
            "Silhouette_Score",
            "Purity"
        ]
        
        # ArcFace specific metrics
        if mode == "arcface":
            metric_fields.extend([
                "Intra_Class_Var",
                "Inter_Class_Dist",
                "Embedding_Norm_Mean",
                "Embedding_Norm_Std"
            ])
        
        self.fieldnames = base_fields + metric_fields
        
        if not self.log_path.exists():
            with self.log_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def log(self, epoch: int, split: str, metrics: Dict[str, float]):
        """Log metrics for a given epoch and split."""
        row = {"epoch": epoch, "split": split}
        for key in self.fieldnames[2:]:  # Skip epoch and split
            row[key] = metrics.get(key, "")
        
        with self.log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class InstantMetricsLogger:
    """
    Unified instant metrics logger for training.
    Only logs lightweight metrics (loss, lr, grad_norm, feature_std, etc.)
    No kNN-Acc or expensive metrics.
    """
    def __init__(self, log_path: Path, mode: str = "pretrain"):
        self.log_path = log_path
        self.mode = mode
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Base fields (shared)
        base_fields = ["iteration", "epoch", "step"]
        
        # Loss fields
        if mode == "pretrain":
            loss_fields = ["loss", "global_loss", "local_loss", "mask_loss", "lr"]
        else:  # finetune or attention query
            loss_fields = ["loss", "lr"]
        
        # Instant monitoring fields (lightweight only)
        instant_fields = [
            "grad_norm", 
            "feature_std", 
            "embedding_norm_mean", 
            "cls_token_norm_mean"
        ]
        
        # Add teacher_center_norm only for pretrain
        if mode == "pretrain":
            instant_fields.extend(["teacher_center_norm", "cosine_similarity"])
        
        self.fieldnames = base_fields + loss_fields + instant_fields
        
        if not self.log_path.exists():
            with self.log_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def log(self, **kwargs):
        """Log all metrics in one row."""
        row = {k: kwargs.get(k, "") for k in self.fieldnames}
        with self.log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
    
    def plot(self, out_dir: Path):
        """Plot training curves with comprehensive metrics (3x3 layout)."""
        if not self.log_path.exists():
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Warning] matplotlib missing -> skip metric plot.")
            return
        
        # Read instant metrics
        df_instant = pd.read_csv(self.log_path)
        if df_instant.empty:
            return
        
        # Read comprehensive metrics (metrics.pretrain.csv)
        metrics_csv = self.log_path.parent / "metrics.pretrain.csv"
        has_metrics = metrics_csv.exists()
        if has_metrics:
            df_metrics = pd.read_csv(metrics_csv)
            # Filter only rows with epoch values (skip empty rows)
            df_metrics = df_metrics[df_metrics['epoch'].notna()]
        
        # Determine available instant metrics
        available_instant = {}
        for col in df_instant.columns:
            if col not in ["iteration", "epoch", "step"]:
                if df_instant[col].notna().any() and (df_instant[col] != "").any():
                    available_instant[col] = True
        
        # Create 3x3 subplot layout
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # ========== Row 1: Instant metrics (original 4 plots, use first 4 positions) ==========
        
        # Plot 0: Losses (always show if loss exists)
        if "loss" in available_instant:
            ax = axes[plot_idx]
            if "loss" in available_instant:
                ax.plot(df_instant["iteration"], df_instant["loss"], label="Total loss", linewidth=1.5)
            if "global_loss" in available_instant:
                ax.plot(df_instant["iteration"], df_instant["global_loss"], label="Global loss", alpha=0.8)
            if "local_loss" in available_instant:
                ax.plot(df_instant["iteration"], df_instant["local_loss"], label="Local loss", alpha=0.8)
            if "mask_loss" in available_instant:
                ax.plot(df_instant["iteration"], df_instant["mask_loss"], label="Mask loss", alpha=0.8)
            ax.set_yscale('log')
            ax.set_xlabel("Iteration", fontsize=10)
            ax.set_ylabel("Loss (log scale)", fontsize=10)
            ax.set_title("Training Losses", fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 1: Cosine Similarity & Gradient Norm
        if "cosine_similarity" in available_instant or "grad_norm" in available_instant:
            ax = axes[plot_idx]
            
            has_cosine = "cosine_similarity" in available_instant
            has_grad = "grad_norm" in available_instant
            
            if has_cosine and has_grad:
                ax_twin = ax.twinx()
                df_filtered = df_instant[df_instant["cosine_similarity"] != ""]
                if not df_filtered.empty:
                    ax.plot(df_filtered["iteration"], df_filtered["cosine_similarity"], 
                           label="Cosine Sim", color='tab:blue', linewidth=1.5)
                    ax.set_ylabel("Cosine Similarity", fontsize=10, color='tab:blue')
                    ax.tick_params(axis='y', labelcolor='tab:blue')
                
                df_filtered = df_instant[df_instant["grad_norm"] != ""]
                if not df_filtered.empty:
                    ax_twin.plot(df_filtered["iteration"], df_filtered["grad_norm"], 
                               label="Grad Norm", color='tab:orange', linewidth=1.5, alpha=0.7)
                    ax_twin.set_ylabel("Gradient Norm", fontsize=10, color='tab:orange')
                    ax_twin.tick_params(axis='y', labelcolor='tab:orange')
                ax.set_title("Cosine Similarity & Gradient Norm", fontsize=12, fontweight='bold')
            
            elif has_cosine:
                df_filtered = df_instant[df_instant["cosine_similarity"] != ""]
                if not df_filtered.empty:
                    ax.plot(df_filtered["iteration"], df_filtered["cosine_similarity"], 
                           label="Cosine Sim", color='tab:blue', linewidth=1.5)
                    ax.set_ylabel("Cosine Similarity", fontsize=10)
                ax.set_title("Cosine Similarity", fontsize=12, fontweight='bold')
            
            else:
                df_filtered = df_instant[df_instant["grad_norm"] != ""]
                if not df_filtered.empty:
                    ax.plot(df_filtered["iteration"], df_filtered["grad_norm"], 
                           label="Grad Norm", color='tab:orange', linewidth=1.5)
                    ax.set_ylabel("Gradient Norm", fontsize=10)
                ax.set_title("Gradient Norm", fontsize=12, fontweight='bold')
            
            ax.set_xlabel("Iteration", fontsize=10)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 2: Feature Statistics
        if any(k in available_instant for k in ["feature_std", "embedding_norm_mean", "cls_token_norm_mean"]):
            ax = axes[plot_idx]
            if "feature_std" in available_instant:
                df_filtered = df_instant[df_instant["feature_std"] != ""]
                if not df_filtered.empty:
                    ax.plot(df_filtered["iteration"], df_filtered["feature_std"], 
                           label="Feature Std", linewidth=1.5)
            if "embedding_norm_mean" in available_instant:
                df_filtered = df_instant[df_instant["embedding_norm_mean"] != ""]
                if not df_filtered.empty:
                    ax.plot(df_filtered["iteration"], df_filtered["embedding_norm_mean"], 
                           label="Embedding Norm", linewidth=1.5, alpha=0.8)
            if "cls_token_norm_mean" in available_instant:
                df_filtered = df_instant[df_instant["cls_token_norm_mean"] != ""]
                if not df_filtered.empty:
                    ax.plot(df_filtered["iteration"], df_filtered["cls_token_norm_mean"], 
                           label="CLS Token Norm", linewidth=1.5, alpha=0.6)
            ax.set_xlabel("Iteration", fontsize=10)
            ax.set_ylabel("Norm / Std", fontsize=10)
            ax.set_title("Feature Statistics", fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Plot 3: Teacher Center Norm & Learning Rate
        if "teacher_center_norm" in available_instant or "lr" in available_instant:
            ax = axes[plot_idx]
            
            has_teacher = "teacher_center_norm" in available_instant
            has_lr = "lr" in available_instant
            
            if has_teacher and has_lr:
                ax_twin = ax.twinx()
                df_filtered = df_instant[df_instant["teacher_center_norm"] != ""]
                if not df_filtered.empty:
                    ax.plot(df_filtered["iteration"], df_filtered["teacher_center_norm"], 
                           label="Teacher Center", color='tab:purple', linewidth=1.5)
                    ax.set_ylabel("Teacher Center Norm", fontsize=10, color='tab:purple')
                    ax.tick_params(axis='y', labelcolor='tab:purple')
                
                ax_twin.plot(df_instant["iteration"], df_instant["lr"], 
                            label="Learning Rate", color='tab:red', linewidth=1.5, alpha=0.7)
                ax_twin.set_ylabel("Learning Rate", fontsize=10, color='tab:red')
                ax_twin.tick_params(axis='y', labelcolor='tab:red')
                ax_twin.set_yscale('log')
                ax.set_title("Teacher Center & Learning Rate", fontsize=12, fontweight='bold')
            
            elif has_teacher:
                df_filtered = df_instant[df_instant["teacher_center_norm"] != ""]
                if not df_filtered.empty:
                    ax.plot(df_filtered["iteration"], df_filtered["teacher_center_norm"], 
                           label="Teacher Center", color='tab:purple', linewidth=1.5)
                    ax.set_ylabel("Teacher Center Norm", fontsize=10)
                ax.set_title("Teacher Center Norm", fontsize=12, fontweight='bold')
            
            else:
                ax.plot(df_instant["iteration"], df_instant["lr"], 
                       label="Learning Rate", color='tab:red', linewidth=1.5)
                ax.set_ylabel("Learning Rate", fontsize=10)
                ax.set_yscale('log')
                ax.set_title("Learning Rate", fontsize=12, fontweight='bold')
            
            ax.set_xlabel("Iteration", fontsize=10)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # ========== Row 2 & 3: Comprehensive metrics (if available) ==========
        
        if has_metrics and not df_metrics.empty:
            # Separate train/test data
            df_train = df_metrics[df_metrics['split'] == 'train']
            df_test = df_metrics[df_metrics['split'] == 'test']
            
            # Plot 4: Recall@1/5/10
            ax = axes[plot_idx]
            if not df_train.empty:
                for k in ['Recall@1', 'Recall@5', 'Recall@10']:
                    if k in df_train.columns:
                        valid_data = df_train[df_train[k] != ""]
                        if not valid_data.empty:
                            ax.plot(valid_data['epoch'], valid_data[k].astype(float), 
                                   marker='o', label=f"{k} (train)", linewidth=1.5)
            if not df_test.empty:
                for k in ['Recall@1', 'Recall@5', 'Recall@10']:
                    if k in df_test.columns:
                        valid_data = df_test[df_test[k] != ""]
                        if not valid_data.empty:
                            ax.plot(valid_data['epoch'], valid_data[k].astype(float), 
                                   marker='s', linestyle='--', label=f"{k} (test)", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Recall", fontsize=10)
            ax.set_title("Recall@K", fontsize=12, fontweight='bold')
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
            # Plot 5: kNN Accuracy k1/5/20
            ax = axes[plot_idx]
            if not df_train.empty:
                for k in ['kNN_Acc_k1', 'kNN_Acc_k5', 'kNN_Acc_k20']:
                    if k in df_train.columns:
                        valid_data = df_train[df_train[k] != ""]
                        if not valid_data.empty:
                            ax.plot(valid_data['epoch'], valid_data[k].astype(float), 
                                   marker='o', label=f"{k} (train)", linewidth=1.5)
            if not df_test.empty:
                for k in ['kNN_Acc_k1', 'kNN_Acc_k5', 'kNN_Acc_k20']:
                    if k in df_test.columns:
                        valid_data = df_test[df_test[k] != ""]
                        if not valid_data.empty:
                            ax.plot(valid_data['epoch'], valid_data[k].astype(float), 
                                   marker='s', linestyle='--', label=f"{k} (test)", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Accuracy", fontsize=10)
            ax.set_title("kNN Accuracy", fontsize=12, fontweight='bold')
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
            # Plot 6: NMI & ARI
            ax = axes[plot_idx]
            if not df_train.empty:
                for k in ['NMI', 'ARI']:
                    if k in df_train.columns:
                        valid_data = df_train[df_train[k] != ""]
                        if not valid_data.empty:
                            ax.plot(valid_data['epoch'], valid_data[k].astype(float), 
                                   marker='o', label=f"{k} (train)", linewidth=1.5)
            if not df_test.empty:
                for k in ['NMI', 'ARI']:
                    if k in df_test.columns:
                        valid_data = df_test[df_test[k] != ""]
                        if not valid_data.empty:
                            ax.plot(valid_data['epoch'], valid_data[k].astype(float), 
                                   marker='s', linestyle='--', label=f"{k} (test)", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title("NMI & ARI", fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
            # Plot 7: Linear Probing / mAP / Purity
            ax = axes[plot_idx]
            if not df_train.empty:
                for k in ['Linear_Probing_Acc', 'mAP', 'Purity']:
                    if k in df_train.columns:
                        valid_data = df_train[df_train[k] != ""]
                        if not valid_data.empty:
                            ax.plot(valid_data['epoch'], valid_data[k].astype(float), 
                                   marker='o', label=f"{k} (train)", linewidth=1.5)
            if not df_test.empty:
                for k in ['Linear_Probing_Acc', 'mAP', 'Purity']:
                    if k in df_test.columns:
                        valid_data = df_test[df_test[k] != ""]
                        if not valid_data.empty:
                            ax.plot(valid_data['epoch'], valid_data[k].astype(float), 
                                   marker='s', linestyle='--', label=f"{k} (test)", linewidth=1.5, alpha=0.7)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title("Linear Probing / mAP / Purity", fontsize=12, fontweight='bold')
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
            # Plot 8: Silhouette Score
            ax = axes[plot_idx]
            if not df_train.empty and 'Silhouette_Score' in df_train.columns:
                valid_data = df_train[df_train['Silhouette_Score'] != ""]
                if not valid_data.empty:
                    ax.plot(valid_data['epoch'], valid_data['Silhouette_Score'].astype(float), 
                           marker='o', label="Train", linewidth=1.5, color='tab:green')
            if not df_test.empty and 'Silhouette_Score' in df_test.columns:
                valid_data = df_test[df_test['Silhouette_Score'] != ""]
                if not valid_data.empty:
                    ax.plot(valid_data['epoch'], valid_data['Silhouette_Score'].astype(float), 
                           marker='s', linestyle='--', label="Test", linewidth=1.5, alpha=0.7, color='tab:orange')
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Silhouette Score", fontsize=10)
            ax.set_title("Silhouette Score", fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, 9):
            axes[idx].axis('off')
        
        fig.tight_layout()
        
        out_path = out_dir / f"training_curves_{self.mode}.pdf"
        fig.savefig(out_path, dpi=300, format="pdf", bbox_inches='tight')
        plt.close(fig)
        print(f"[Info] Saved training curves to {out_path}")


# -------------------------------------------------------
# Metrics Computation Functions
# -------------------------------------------------------

def compute_recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Compute Recall@K for retrieval tasks.
    
    Args:
        embeddings: [N, D] normalized embeddings
        labels: [N] ground truth labels (can be list or array)
        k_values: list of K values
    
    Returns:
        Dictionary with Recall@K scores
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize
    except ImportError:
        return {f"Recall@{k}": 0.0 for k in k_values}
    
    # Convert labels to numpy array if it's a list
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    if len(embeddings) == 0 or len(labels) == 0:
        return {f"Recall@{k}": 0.0 for k in k_values}
    
    embeddings = normalize(embeddings, norm='l2')
    similarity_matrix = cosine_similarity(embeddings)
    
    # Set diagonal to -inf to exclude self-similarity
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    recalls = {}
    for k in k_values:
        k_actual = min(k, len(labels) - 1)
        if k_actual < 1:
            recalls[f"Recall@{k}"] = 0.0
            continue
        
        correct = 0
        for i in range(len(labels)):
            # Get top-k most similar samples
            top_k_indices = np.argsort(similarity_matrix[i])[-k_actual:]
            top_k_labels = labels[top_k_indices]
            
            # Check if any of top-k has the same label
            if labels[i] in top_k_labels:
                correct += 1
        
        recalls[f"Recall@{k}"] = correct / len(labels)
    
    return recalls


def compute_knn_accuracy(embeddings: np.ndarray, labels: np.ndarray, k_values: List[int] = [1, 5, 20]) -> Dict[str, float]:
    """
    Compute k-NN classification accuracy.
    
    Args:
        embeddings: [N, D] normalized embeddings
        labels: [N] ground truth labels
        k_values: list of K values for k-NN
    
    Returns:
        Dictionary with kNN accuracy for each k
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import normalize
    except ImportError:
        return {f"kNN_Acc_k{k}": 0.0 for k in k_values}
    
    embeddings = normalize(embeddings, norm='l2')
    
    accuracies = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        # Use 5-fold cross-validation
        scores = cross_val_score(knn, embeddings, labels, cv=min(5, len(np.unique(labels))))
        accuracies[f"kNN_Acc_k{k}"] = scores.mean()
    
    return accuracies


def compute_linear_probing_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Train a linear classifier on frozen embeddings (linear probing).
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] ground truth labels
    
    Returns:
        Linear probing accuracy
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import normalize
    except ImportError:
        return 0.0
    
    embeddings = normalize(embeddings, norm='l2')
    
    # Use L2-regularized logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, embeddings, labels, cv=min(5, len(np.unique(labels))))
    
    return scores.mean()


def compute_mean_average_precision(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute mean Average Precision (mAP) for retrieval.
    
    Args:
        embeddings: [N, D] normalized embeddings
        labels: [N] ground truth labels (can be list or array)
    
    Returns:
        mAP score
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize
    except ImportError:
        return 0.0
    
    # Convert labels to numpy array if it's a list
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    embeddings = normalize(embeddings, norm='l2')
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, -np.inf)
    
    aps = []
    for i in range(len(labels)):
        # Get sorted indices by similarity
        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        sorted_labels = labels[sorted_indices] 
        
        # Compute Average Precision for this query
        relevant = (sorted_labels == labels[i])
        if relevant.sum() == 0:
            continue
        
        precision_at_k = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = (precision_at_k * relevant).sum() / relevant.sum()
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0


def compute_silhouette_score(embeddings: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
    """
    Compute Silhouette Score (can work without labels using clustering).
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] ground truth labels (optional, will use KMeans if None)
    
    Returns:
        Silhouette score
    """
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
    except ImportError:
        return 0.0
    
    if len(embeddings) < 2:
        return 0.0
    
    embeddings = normalize(embeddings, norm='l2')
    
    if labels is None:
        # Estimate number of clusters (heuristic: sqrt(n/2))
        n_clusters = max(2, int(np.sqrt(len(embeddings) / 2)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
    
    # Need at least 2 clusters
    if len(np.unique(labels)) < 2:
        return 0.0
    
    return silhouette_score(embeddings, labels, metric='cosine')


def compute_purity(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute clustering purity using K-Means.
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] ground truth labels (can be list or array)
    
    Returns:
        Purity score
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
    except ImportError:
        return 0.0
    
    # Convert labels to numpy array if it's a list
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    embeddings = normalize(embeddings, norm='l2')
    
    # Get unique labels and create label mapping
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    
    # Create label to index mapping
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert labels to indices
    label_indices = np.array([label_to_idx[label] for label in labels])
    
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute purity
    contingency_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(labels)):
        contingency_matrix[cluster_labels[i], label_indices[i]] += 1
    
    purity = np.sum(np.max(contingency_matrix, axis=1)) / len(labels)
    return purity


def compute_arcface_specific_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Compute ArcFace-specific metrics: intra-class variance and inter-class distance.
    
    Args:
        embeddings: [N, D] normalized embeddings
        labels: [N] ground truth labels (can be list or array, should be numeric indices)
    
    Returns:
        Dictionary with intra/inter class metrics
    """
    try:
        from sklearn.preprocessing import normalize
        from sklearn.metrics.pairwise import euclidean_distances
    except ImportError:
        return {
            "Intra_Class_Var": 0.0,
            "Inter_Class_Dist": 0.0,
            "Embedding_Norm_Mean": 0.0,
            "Embedding_Norm_Std": 0.0
        }
    
    # Convert labels to numpy array if it's a list
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    embeddings = normalize(embeddings, norm='l2')
    unique_labels = np.unique(labels)
    
    # Intra-class variance
    intra_vars = []
    class_centers = []
    for label in unique_labels:
        class_embeddings = embeddings[labels == label]
        if len(class_embeddings) > 1:
            center = class_embeddings.mean(axis=0)
            class_centers.append(center)
            variance = np.mean(np.sum((class_embeddings - center) ** 2, axis=1))
            intra_vars.append(variance)
    
    intra_class_var = np.mean(intra_vars) if intra_vars else 0.0
    
    # Inter-class distance
    if len(class_centers) > 1:
        class_centers = np.array(class_centers)
        distances = euclidean_distances(class_centers)
        # Get upper triangle (exclude diagonal)
        inter_class_dist = distances[np.triu_indices_from(distances, k=1)].mean()
    else:
        inter_class_dist = 0.0
    
    # Embedding norm statistics
    norms = np.linalg.norm(embeddings, axis=1)
    
    return {
        "Intra_Class_Var": intra_class_var,
        "Inter_Class_Dist": inter_class_dist,
        "Embedding_Norm_Mean": norms.mean(),
        "Embedding_Norm_Std": norms.std()
    }


def maybe_subsample_for_metrics(embeddings, labels, max_samples, seed=42):
    embeddings = np.asarray(embeddings)
    n = embeddings.shape[0]
    if max_samples is None or max_samples <= 0 or n <= max_samples:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_samples, replace=False))
    subset_embeddings = embeddings[idx]
    subset_labels = None
    if labels is not None:
        labels_array = np.array(labels)
        subset_labels = labels_array[idx]
        if isinstance(labels, list):
            subset_labels = subset_labels.tolist()
    print(f"[Info] Metrics subsample: {len(idx)}/{n} samples (seed=42)")
    return subset_embeddings, subset_labels


def compute_all_metrics(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray],
    mode: str = "pretrain",
    compute_linear_probing: bool = False
) -> Dict[str, float]:
    """
    Compute all metrics at once.
    
    Args:
        embeddings: [N, D] embeddings
        labels: [N] ground truth labels (None for unlabeled data, can be list or array)
        mode: "pretrain" or "arcface"
        compute_linear_probing: whether to compute expensive linear probing
    
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Convert labels to numpy array if it's a list
    if labels is not None and not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    if len(embeddings) == 0:
        print("[Warning] Empty embeddings, skipping metrics computation")
        return {k: "" for k in ["NMI", "ARI", "Recall@1", "Recall@5", "Recall@10",
                                "kNN_Acc_k1", "kNN_Acc_k5", "kNN_Acc_k20", "Linear_Probing_Acc", 
                                "mAP", "Purity", "Silhouette_Score"]}
    
    try:
        from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize
    
        embeddings_norm = normalize(embeddings, norm='l2')
    
        if labels is None or len(labels) == 0:
            print("[Info] No labels provided, using KMeans clustering for unsupervised metrics")
            n_clusters = max(2, int(np.sqrt(len(embeddings) / 2))) 
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pseudo_labels = kmeans.fit_predict(embeddings_norm)
        
            # unsuperwised metrics (pseudolabels from clustering)
            metrics["Silhouette_Score"] = compute_silhouette_score(embeddings, pseudo_labels)
            
            for key in ["NMI", "ARI", "Recall@1", "Recall@5", "Recall@10", "kNN_Acc_k1",
                       "kNN_Acc_k5", "kNN_Acc_k20", "Linear_Probing_Acc", "mAP", "Purity"]:
                metrics[key] = ""
        
            print(f"[Metrics] Unsupervised metrics computed (estimated {n_clusters} clusters)")
    
        else:
            # Get unique labels and create label mapping for clustering
            unique_labels = np.unique(labels)
            n_classes = len(unique_labels)
        
            # Create label to index mapping
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            label_indices = np.array([label_to_idx[label] for label in labels])
        
            kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
            pred_labels = kmeans.fit_predict(embeddings_norm)
        
            # NMI 和 ARI (supervised)
            metrics["NMI"] = normalized_mutual_info_score(label_indices, pred_labels)
            metrics["ARI"] = adjusted_rand_score(label_indices, pred_labels)
        
            # Silhouette Score (with true labels)
            metrics["Silhouette_Score"] = compute_silhouette_score(embeddings, label_indices)
        
            # Recall@K
            if len(embeddings) >= 2:
                recall_metrics = compute_recall_at_k(embeddings, labels, k_values=[1, 5, 10])
                metrics.update(recall_metrics)
            else:
                print("[Warning] Not enough samples for Recall@K")
                metrics.update({"Recall@1": "", "Recall@5": "", "Recall@10": ""})
        
            # kNN Accuracy
            if len(embeddings) >= n_classes * 2:
                knn_metrics = compute_knn_accuracy(embeddings, label_indices, k_values=[1, 5, 20])
                metrics.update(knn_metrics)
            else:
                print("[Warning] Not enough samples for kNN")
                metrics.update({"kNN_Acc_k1": "", "kNN_Acc_k5": "", "kNN_Acc_k20": ""})
        
            # Linear Probing (expensive, compute less frequently)
            if compute_linear_probing and len(embeddings) >= n_classes * 2:
                metrics["Linear_Probing_Acc"] = compute_linear_probing_accuracy(embeddings, label_indices)
            else:
                metrics["Linear_Probing_Acc"] = ""
        
            # mAP
            if len(embeddings) >= 2:
                metrics["mAP"] = compute_mean_average_precision(embeddings, labels)
            else:
                metrics["mAP"] = ""
        
            # Purity
            if len(embeddings) >= n_classes:
                metrics["Purity"] = compute_purity(embeddings, labels)
            else:
                metrics["Purity"] = ""
        
            # ArcFace specific metrics
            if mode == "arcface":
                arcface_metrics = compute_arcface_specific_metrics(embeddings, label_indices)
                metrics.update(arcface_metrics)
    except Exception as e:
        print(f"[Warning] Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        # Fill with empty values
        for key in ["NMI", "ARI", "Recall@1", "Recall@5", "Recall@10", "kNN_Acc_k1",
                   "kNN_Acc_k5", "kNN_Acc_k20", "Linear_Probing_Acc", "mAP", "Purity", "Silhouette_Score"]:
            if key not in metrics:
                metrics[key] = ""
    return metrics


def compute_instant_metrics(model, batch_data, device) -> Dict[str, float]:
    """
    Compute instant metrics during training (gradient norm, feature std, etc.)
    """
    metrics = {}
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    metrics["grad_norm"] = total_norm ** 0.5
    
    with torch.no_grad():
        views = [v.to(device) for v in batch_data["views"][:2]]
        
        backbone_feats = model.backbone.forward_features(views[0])
        if isinstance(backbone_feats, dict):
            tokens = backbone_feats["x"]  # [B, N+1, D]
        else:
            tokens = backbone_feats
        
        cls_token_raw = tokens[:, 0]  # [B, D] - raw CLS token
        proj_output_normalized = model.projector(cls_token_raw)  # Already normalized
        feature_std = cls_token_raw.std(dim=0).mean().item()
        metrics["feature_std"] = feature_std
        embedding_norm = proj_output_normalized.norm(dim=1).mean().item()
        metrics["embedding_norm_mean"] = embedding_norm
        cls_norm = cls_token_raw.norm(dim=1).mean().item()
        metrics["cls_token_norm_mean"] = cls_norm
        
    return metrics


def compute_instant_metrics_arcface(model, batch_data, labels, device) -> Dict[str, float]:
    """
    Compute instant metrics during ArcFace finetuning.
    """
    metrics = {}
    
    # Gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    metrics["grad_norm"] = total_norm ** 0.5
    
    with torch.no_grad():
        # Get raw features
        backbone_feats = model.backbone.forward_features(batch_data)
        if isinstance(backbone_feats, dict):
            tokens = backbone_feats["x"]
        else:
            tokens = backbone_feats
        
        cls_token_raw = tokens[:, 0]  # Unnormalized
        
        # ArcFace projector is Sequential (no built-in normalize)
        proj_output = model.projector(cls_token_raw)
        
        # For ArcFace, embeddings are typically L2-normalized before loss
        proj_output_norm = F.normalize(proj_output, dim=-1)
        
        # Statistics
        feature_std = cls_token_raw.std(dim=0).mean().item()
        metrics["feature_std"] = feature_std
        
        # Normalized embedding norm
        metrics["embedding_norm_mean"] = proj_output_norm.norm(dim=1).mean().item()
        
        # CLS token norm
        metrics["cls_token_norm_mean"] = cls_token_raw.norm(dim=1).mean().item()
        
    return metrics


# -------------------------------------------------------
# Training Functions
# -------------------------------------------------------

def compute_and_log_all_metrics(
    args, model, device, epoch, logs_dir, metrics_logger, mode="pretrain"
):
    """
    Compute and log all metrics for train/test sets.
    
    Args:
        args: arguments
        model: trained model
        device: torch device
        epoch: current epoch
        logs_dir: directory to save logs
        metrics_logger: EnhancedMetricsLogger instance
        mode: "pretrain" or "arcface"
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    transform = build_eval_transform(args.extract_size)
    
    # Determine if we should compute linear probing (expensive)
    compute_lp = (epoch + 1) % 10 == 0
    
    # Train set
    if args.visualize_train_data:
        print(f"[Metrics] Computing metrics for train set at epoch {epoch+1}...")
        feats, files, labels = compute_embeddings_from_csv(
            model=model,
            csv_path=args.train_data,
            root_dir=args.input_images_dir,
            transform=transform,
            device=device,
            use_projector=args.use_projector_output,
            token_mode=args.token_mode,
            topk_patches=args.topk_patches,
        )
        feats_eval, labels_eval = maybe_subsample_for_metrics(
            feats, labels, args.metrics_sample_size
        )
        metrics = compute_all_metrics(feats_eval,labels_eval, mode=mode, compute_linear_probing=compute_lp)
        metrics_logger.log(epoch + 1, "train", metrics)
        
        # Print summary
        print(f"[Metrics] Epoch {epoch+1} Train:")
        for k, v in metrics.items():
            if v != "":
                print(f"  {k}: {v:.4f}")
        
        # UMAP visualization
        if labels is not None:
            run_umap_wrapper(
                features=feats,
                labels=labels,
                out_dir=logs_dir,
                filename=f"umap.train.epoch_{epoch+1:04d}.pdf",
                title=f"UMAP Train - Epoch {epoch+1}",
                args=args,
            )
    
    # Test set
    if args.visualize_test_data:
        print(f"[Metrics] Computing metrics for test set at epoch {epoch+1}...")
        feats, files, labels = compute_embeddings_from_csv(
            model=model,
            csv_path=args.visualize_test_data,
            root_dir=args.input_images_dir,
            transform=transform,
            device=device,
            use_projector=args.use_projector_output,
            token_mode=args.token_mode,
            topk_patches=args.topk_patches,
        )
        feats_eval, labels_eval = maybe_subsample_for_metrics(
            feats, labels, args.metrics_sample_size
        )        
        metrics = compute_all_metrics(feats_eval, labels_eval, mode=mode, compute_linear_probing=compute_lp)
        metrics_logger.log(epoch + 1, "test", metrics)
        
        # Print summary
        print(f"[Metrics] Epoch {epoch+1} Test:")
        for k, v in metrics.items():
            if v != "":
                print(f"  {k}: {v:.4f}")
        
        # UMAP visualization
        if labels is not None:
            run_umap_wrapper(
                features=feats,
                labels=labels,
                out_dir=logs_dir,
                filename=f"umap.test.epoch_{epoch+1:04d}.pdf",
                title=f"UMAP Test - Epoch {epoch+1}",
                args=args,
            )


# -------------------------------------------------------
# Utilities 
# -------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    device_str = device_str.lower()
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_num_workers(num_workers: int) -> int:
    if num_workers < 0:
        return max(1, os.cpu_count() or 1)
    return num_workers


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_iters = warmup_epochs * niter_per_ep
    iters = np.arange(epochs * niter_per_ep)
    schedule = np.ones_like(iters) * final_value
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        schedule[:warmup_iters] = warmup_schedule
    iters_after_warmup = iters[warmup_iters:]
    if len(iters_after_warmup) > 0:
        schedule[warmup_iters:] = final_value + 0.5 * (base_value - final_value) * \
            (1 + np.cos(np.pi * (iters_after_warmup - warmup_iters) / (len(iters_after_warmup))))
    return schedule


def auto_find_checkpoint(out_dir: Path, mode: str) -> str:
    """Auto-find checkpoint for finetune/extract modes."""
    if mode == "finetune":
        arcface_latest = out_dir / "arcface_latest.pth"
        if arcface_latest.exists():
            print(f"[Info] Found existing ArcFace checkpoint, will resume from: {arcface_latest}")
            return str(arcface_latest)
        
        SSL_latest = out_dir / "SSL_latest.pth"
        if SSL_latest.exists():
            print(f"[Info] Auto-detected SSL pretrain checkpoint: {SSL_latest}")
            return str(SSL_latest)
        
        raise FileNotFoundError(
            f"No pretrained checkpoint found in {out_dir}.\n"
            f"Please run '--mode pretrain' first, or specify --checkpoint manually."
        )
    
    elif mode == "extract":
        arcface_latest = out_dir / "arcface_latest.pth"
        if arcface_latest.exists():
            print(f"[Info] Using ArcFace finetuned checkpoint: {arcface_latest}")
            return str(arcface_latest)
        
        SSL_latest = out_dir / "SSL_latest.pth"
        if SSL_latest.exists():
            print(f"[Info] Using SSL pretrained checkpoint: {SSL_latest}")
            return str(SSL_latest)
        
        raise FileNotFoundError(
            f"No checkpoint found in {out_dir}.\n"
            f"Please run '--mode pretrain' or '--mode finetune' first."
        )
    
    return ""


# -------------------------------------------------------
# Datasets & transforms
# -------------------------------------------------------

class MultiCropDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        transform_global,
        transform_local,
        local_crops: int,
    ):
        meta = pd.read_csv(csv_path)
        if "image" not in meta.columns:
            raise ValueError("train_data CSV must contain an 'image' column.")
        self.paths = meta["image"].tolist()
        self.labels = meta["label"].tolist() if "label" in meta.columns else None
        self.root_dir = Path(root_dir)
        self.transform_global = transform_global
        self.transform_local = transform_local
        self.local_crops = local_crops

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        img_path = self.root_dir / rel_path
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        view1 = self.transform_global(img)
        view2 = self.transform_global(img)
        locals_list = [self.transform_local(img) for _ in range(self.local_crops)]
        
        item = {"index": idx, "views": [view1, view2] + locals_list}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class CSVImageDataset(Dataset):
    def __init__(self, csv_path: str, root_dir: str, transform, require_label: bool = False):
        meta = pd.read_csv(csv_path)
        if "image" not in meta.columns:
            raise ValueError(f"{csv_path} must contain an 'image' column.")
        self.paths = meta["image"].tolist()
        self.labels = meta["label"].tolist() if "label" in meta.columns else None
        if require_label and self.labels is None:
            raise ValueError(f"{csv_path} must contain a 'label' column.")
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        rel = self.paths[idx]
        img_path = self.root_dir / rel
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        tensor = self.transform(img)
        if self.labels is None:
            return tensor, rel, None
        return tensor, rel, self.labels[idx]


class FolderDataset(Dataset):
    def __init__(self, folder: str, transform):
        self.paths = sorted([p for p in Path(folder).rglob("*")
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), str(path)


def build_transforms(global_size: int, local_size: int):
    """Build training transforms with rotation and flip augmentation."""
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    transform_global = transforms.Compose([
        transforms.RandomResizedCrop(
            global_size, 
            scale=(0.4, 1.0),
            interpolation=Image.BICUBIC
        ),
        transforms.RandomRotation(
            degrees=180,
            interpolation=Image.BICUBIC,
            expand=False,
            fill=0
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        normalize,
    ])
    
    transform_local = transforms.Compose([
        transforms.RandomResizedCrop(
            local_size, 
            scale=(0.05, 0.4), 
            interpolation=Image.BICUBIC
        ),
        transforms.RandomRotation(
            degrees=180,
            interpolation=Image.BICUBIC,
            expand=False,
            fill=0
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        normalize,
    ])
    
    return transform_global, transform_local


def build_eval_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])


# -------------------------------------------------------
# SSL modules
# -------------------------------------------------------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

class SSLModel(nn.Module):
    def __init__(
        self, 
        model_name: str = "vit_small_patch16_224", 
        out_dim: int = 256, 
        img_size: int = 224,
        use_attention_pooling: bool = False,
        pooling_type: str = "lightweight"
    ):
        super().__init__()
        self.model_name = model_name

        # check patch size
        patch_size = 16
        if 'patch14' in model_name or 'eva' in model_name.lower():
            patch_size = 14
        elif 'patch32' in model_name:
            patch_size = 32
        
        if img_size % patch_size != 0:
            aligned_size = (img_size // patch_size) * patch_size
            print(f"[Warning] img_size {img_size} not divisible by patch_size {patch_size}, "
                  f"auto-aligned to {aligned_size}")
            img_size = aligned_size
        
        self.img_size = img_size

        try:
            # ViT-like models
            self.backbone = timm.create_model(
                model_name,
                img_size=img_size,
                num_classes=0,
                global_pool="",
                pretrained=True,
                dynamic_img_size=True,
            )
        except Exception as e:
            # EVA-like models
            self.backbone = timm.create_model(
                model_name,
                img_size=img_size,
                pretrained=True,
                dynamic_img_size=True,
            )
            if hasattr(self.backbone, 'head'):
                self.backbone.head = nn.Identity()
            if hasattr(self.backbone, 'fc_norm'):
                self.backbone.fc_norm = nn.Identity()
        embed_dim = self.backbone.num_features

        # attention pooling
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            if pooling_type == "lightweight":
                self.attention_pool = LightweightAttentionPooling(embed_dim)
            elif pooling_type == "multihead":
                self.attention_pool = MultiHeadAttentionPooling(embed_dim, freeze_attn=True)
            elif pooling_type == "gated":
                self.attention_pool = GatedAttentionPooling(embed_dim, hidden_dim=128)
            else:
                raise ValueError(f"Unknown pooling_type: {pooling_type}")
        
        self.projector = ProjectionHead(embed_dim, out_dim)
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, x, return_tokens: bool = False, return_all_patches: bool = False):
        feats = self.backbone.forward_features(x)
        tokens = feats["x"] if isinstance(feats, dict) else feats
        
        # Use attention pooling if enabled
        if self.use_attention_pooling:
            patch_tokens = tokens[:, 1:, :]
            pooled_features, attn_weights = self.attention_pool(patch_tokens)
            proj = self.projector(pooled_features)
            
            if return_all_patches:
                return proj, tokens
            if return_tokens:
                return proj, patch_tokens
            return proj, pooled_features
        
        # CLS token
        cls_token = tokens[:, 0]
        proj = self.projector(cls_token)
        if return_all_patches:
            return proj, tokens
        if return_tokens:
            return proj, tokens[:, 1:]
        return proj, cls_token

    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center with exponential moving average."""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center = self.center * 0.9 + batch_center * 0.1



def ema_update(model_teacher: nn.Module, model_student: nn.Module, momentum: float):
    with torch.no_grad():
        for param_t, param_s in zip(model_teacher.parameters(), model_student.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=(1.0 - momentum))


def ssl_loss(student_out, teacher_out, temp_student, temp_teacher):
    """Contrastive loss using batch-wise similarity matrix."""
    B = student_out.size(0)
    student_sim = student_out @ teacher_out.T / temp_student
    
    with torch.no_grad():
        teacher_sim = teacher_out @ teacher_out.T / temp_teacher
        teacher_sim = teacher_sim - teacher_sim.mean(dim=1, keepdim=True)
        teacher_probs = F.softmax(teacher_sim, dim=1)
    
    student_log_probs = F.log_softmax(student_sim, dim=1)
    loss = -(teacher_probs * student_log_probs).sum(dim=1).mean()
    
    return loss


def masked_token_loss(student_tokens, teacher_tokens, mask_ratio=0.4, model_name=""):
    """
    Masked token reconstruction loss.
    - ViT models: cosine distance (normalized)
    - EVA models: raw L2 distance (magnitude-sensitive)
    """
    B, N, C = student_tokens.shape
    num_mask = max(1, int(mask_ratio * N))
    
    mask_indices = torch.rand(B, N, device=student_tokens.device).argsort(dim=1)[:, :num_mask]
    
    student_masked = torch.gather(
        student_tokens, 
        dim=1, 
        index=mask_indices.unsqueeze(-1).expand(-1, -1, C)
    )
    
    teacher_masked = torch.gather(
        teacher_tokens, 
        dim=1, 
        index=mask_indices.unsqueeze(-1).expand(-1, -1, C)
    ).detach()
    
    is_eva = 'eva' in model_name.lower()
    
    if is_eva:
        loss = F.mse_loss(student_masked, teacher_masked)
    else:
        student_masked = F.normalize(student_masked, dim=-1)
        teacher_masked = F.normalize(teacher_masked, dim=-1)
        loss = 2 - 2 * (student_masked * teacher_masked).sum(dim=-1)
        loss = loss.mean()
    
    return loss


def align_to_patch(size, patch_size, direction='nearest'):
    if size % patch_size == 0:
        return size
    if direction == 'nearest':
        return round(size / patch_size) * patch_size
    elif direction == 'up':
        return math.ceil(size / patch_size) * patch_size
    else:
        return (size // patch_size) * patch_size

# -------------------------------------------------------
# Attention Pooling Modules
# -------------------------------------------------------

class LightweightAttentionPooling(nn.Module):
    """
    Lightweight attention pooling with learnable query vector.
    Only 768 parameters - ideal for small datasets.
    """
    def __init__(self, dim=768):
        super().__init__()
        # Learnable global query (only trainable parameter)
        self.query = nn.Parameter(torch.randn(1, dim))
        self.scale = dim ** -0.5
        
    def forward(self, patch_tokens):
        # patch_tokens: [B, N, D]
        B, N, D = patch_tokens.shape
        
        # Expand query to batch
        query = self.query.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
        
        # Compute attention scores
        attn_scores = torch.bmm(query, patch_tokens.transpose(1, 2))  # [B, 1, N]
        attn_scores = attn_scores * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Weighted average
        pooled = torch.bmm(attn_weights, patch_tokens)  # [B, 1, D]
        
        return pooled.squeeze(1), attn_weights.squeeze(1)  # [B, D], [B, N]


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling (can freeze attention weights).
    """
    def __init__(self, dim=768, num_heads=8, freeze_attn=True):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        if freeze_attn:
            # Freeze attention weights, only train query
            for param in self.attn.parameters():
                param.requires_grad = False
        
    def forward(self, patch_tokens):
        B = patch_tokens.size(0)
        query = self.query.expand(B, -1, -1)
        
        attn_output, attn_weights = self.attn(
            query, patch_tokens, patch_tokens,
            need_weights=True
        )
        
        return attn_output.squeeze(1), attn_weights.squeeze(1)


class GatedAttentionPooling(nn.Module):
    """
    Gated attention pooling (ABMIL-style).
    Better for fine-grained features but needs more samples.
    """
    def __init__(self, dim=768, hidden_dim=128):
        super().__init__()
        # Attention branch
        self.attention_V = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        
    def forward(self, patch_tokens):
        # patch_tokens: [B, N, D]
        
        # Compute attention scores with gating
        A_V = self.attention_V(patch_tokens)  # [B, N, hidden_dim]
        A_U = self.attention_U(patch_tokens)  # [B, N, hidden_dim]
        A = self.attention_w(A_V * A_U)  # [B, N, 1] (element-wise gating)
        
        attn_weights = F.softmax(A, dim=1)  # [B, N, 1]
        
        # Weighted average
        pooled = torch.sum(attn_weights * patch_tokens, dim=1)  # [B, D]
        
        return pooled, attn_weights.squeeze(-1)



# -------------------------------------------------------
# Training with Enhanced Metrics
# -------------------------------------------------------

def train_ssl(args):
    # save proceseed CSV with aboulute paths
    output_dir = Path(args.out_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    train_meta = pd.read_csv(args.train_data)
    train_root = Path(args.input_images_dir)
    
    processed_train = train_meta.copy()
    processed_train["image"] = processed_train["image"].apply(
        lambda x: str((train_root / x).expanduser().resolve())
    )
    
    processed_csv_path = output_dir / "train.processed.csv"
    processed_train.to_csv(processed_csv_path, index=False)
    print(f"[Info] Saved processed training CSV to {processed_csv_path}")
    
    if args.visualize_test_data:
        test_meta = pd.read_csv(args.visualize_test_data)
        processed_test = test_meta.copy()
        processed_test["image"] = processed_test["image"].apply(
            lambda x: str((train_root / x).expanduser().resolve())
        )
        test_csv_path = output_dir / "test.processed.csv"
        processed_test.to_csv(test_csv_path, index=False)
        print(f"[Info] Saved processed test CSV to {test_csv_path}")

    # training
    device = resolve_device(args.device)
    set_seed(args.seed)
    
    t_global, t_local = build_transforms(
        global_size=args.global_crop_size,
        local_size=args.local_crop_size,
    )
    dataset = MultiCropDataset(
        csv_path=args.train_data,
        root_dir=args.input_images_dir,
        transform_global=t_global,
        transform_local=t_local,
        local_crops=args.local_crops,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=resolve_num_workers(args.num_workers),
        drop_last=True,
    )
    student = SSLModel(args.model_name, args.out_dim, img_size=args.global_crop_size).to(device)
    
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    
    set_seed(args.seed + 1)
    teacher = SSLModel(args.model_name, args.out_dim, img_size=args.global_crop_size).to(device)
    
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    
    for p in teacher.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Load checkpoint first to get start_epoch and iteration
    start_epoch = 0
    it = 0
    
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["student"])
        teacher.load_state_dict(ckpt["teacher"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        it = ckpt.get("iteration", start_epoch * len(loader))
        print(f"[Info] Resume from {args.resume} at epoch {start_epoch}, iteration {it}")
    
    warmup_epochs = args.warmup_epochs
    warmup_iters = warmup_epochs * len(loader)
    
    # If resuming and already past warmup phase, disable warmup
    if args.resume and it >= warmup_iters:
        print(f"[Info] Already past warmup phase (iter {it} >= {warmup_iters}), setting warmup_epochs=0")
        warmup_epochs = 0
    elif args.resume and it > 0:
        # Partially through warmup - adjust warmup to remaining iterations
        remaining_warmup_iters = warmup_iters - it
        warmup_epochs = remaining_warmup_iters / len(loader)
        print(f"[Info] Resuming during warmup phase, adjusting warmup to {warmup_epochs:.2f} epochs")

    total_iters = len(loader) * args.max_epochs
    
    lr_schedule = cosine_scheduler(
        base_value=args.lr,
        final_value=args.lr * 0.01,
        epochs=args.max_epochs,
        niter_per_ep=len(loader),
        warmup_epochs=warmup_epochs,
        start_warmup_value=args.lr * 0.1,
    )
    
    temp_schedule = np.linspace(args.student_temp, args.student_temp, total_iters)
    warmup_ratio = 0.7
    warmup_iters_teacher = int(total_iters * warmup_ratio)
    teacher_temp_schedule = np.concatenate([
        np.linspace(args.teacher_temp_start, args.teacher_temp_end, warmup_iters_teacher),
        np.ones(total_iters - warmup_iters_teacher) * args.teacher_temp_end
    ])
    
    momentum_schedule = cosine_scheduler(
        base_value=args.teacher_momentum,
        final_value=args.teacher_momentum_end,
        epochs=args.max_epochs,
        niter_per_ep=len(loader),
    )
    output_dir = Path(args.out_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize loggers
    instant_logger = InstantMetricsLogger(logs_dir / "instant_metrics.pretrain.csv", mode="pretrain")
    enhanced_logger = EnhancedMetricsLogger(logs_dir / "metrics.pretrain.csv", mode="pretrain")
    # Check initial similarity
    with torch.no_grad():
        sample_batch = next(iter(loader))
        sample_views = [v.to(device) for v in sample_batch["views"][:2]]
        s_out, _ = student(sample_views[0], return_tokens=True)
        t_out, _ = teacher(sample_views[0], return_tokens=True)
        init_cosine = F.cosine_similarity(s_out.mean(0, keepdim=True), t_out.mean(0, keepdim=True)).item()
        print(f"[Info] Initial cosine similarity: {init_cosine:.4f}")
        
        # Print current learning rate
        current_lr = lr_schedule[it]
        print(f"[Info] Starting/Resuming with learning rate: {current_lr:.6f}")

    for epoch in range(start_epoch, args.max_epochs):
        cosine_sim = 0.0
        student.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.max_epochs}", ncols=150)
        
        for step, batch in enumerate(pbar):
            views = [v.to(device, non_blocking=True) for v in batch["views"]]
            global_views, local_views = views[:2], views[2:]
            # Student forward
            student_global, student_tokens = [], []
            for view in global_views:
                proj, tokens = student(view, return_tokens=True)
                student_global.append(proj)
                student_tokens.append(tokens)
            # Teacher forward
            teacher_global, teacher_tokens = [], []
            with torch.no_grad():
                for view in global_views:
                    proj, tokens = teacher(view, return_tokens=True)
                    proj = proj - teacher.center
                    proj = F.normalize(proj, dim=-1)
                    teacher_global.append(proj)
                    teacher_tokens.append(tokens)
                
                all_teacher = torch.cat(teacher_global, dim=0)
                teacher.update_center(all_teacher)

            temp_student = temp_schedule[it]
            temp_teacher = teacher_temp_schedule[it]
 
            loss_global = 0.0
            if args.disable_cross_view_loss:
                for s, t in zip(student_global, teacher_global):
                    loss_global += ssl_loss(s, t, temp_student, temp_teacher)
                loss_global /= len(student_global)
            else:
                for s in student_global:
                    for t in teacher_global:
                        loss_global += ssl_loss(s, t, temp_student, temp_teacher)
                loss_global /= (len(student_global) * len(teacher_global))

            loss_mask = 0.0
            for s_tokens, t_tokens in zip(student_tokens, teacher_tokens):
                loss_mask += masked_token_loss(s_tokens, t_tokens, args.mask_ratio, args.model_name)
            loss_mask /= len(student_tokens)

            loss_local = 0.0
            if local_views:
                for lv in local_views:
                    proj, _ = student(lv, return_tokens=True)
                    for t in teacher_global:
                        loss_local += ssl_loss(proj, t, temp_student, temp_teacher)
                loss_local /= (len(local_views) * len(teacher_global))

            loss = loss_global + args.lambda_local * loss_local + args.lambda_mask * loss_mask
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            optimizer.step()
            
            current_momentum = momentum_schedule[it]
            ema_update(teacher, student, current_momentum)
            for g in optimizer.param_groups:
                g["lr"] = lr_schedule[it]
            # Logging (only lightweight metrics)
            log_every = max(1, args.log_every_n_steps)
            if it % log_every == 0:
                with torch.no_grad():
                    s_raw, _ = student(global_views[0], return_tokens=True)
                    t_raw, _ = teacher(global_views[0], return_tokens=True)
                    cosine_sim = F.cosine_similarity(
                        s_raw.mean(dim=0, keepdim=True),
                        t_raw.mean(dim=0, keepdim=True)
                    ).item()
                    
                    # Instant metrics (lightweight only)
                    instant_metrics = compute_instant_metrics(student, batch, device)
                    instant_metrics["teacher_center_norm"] = teacher.center.norm().item()
                    instant_metrics["cosine_similarity"] = cosine_sim
                    # Unified logging (loss + instant metrics)
                    instant_logger.log(
                        iteration=it,
                        epoch=epoch,
                        step=step,
                        loss=loss.item(),
                        global_loss=loss_global.item(),
                        local_loss=loss_local.item(),
                        mask_loss=loss_mask.item(),
                        lr=optimizer.param_groups[0]["lr"],
                        teacher_temp=teacher_temp_schedule[it],
                        **instant_metrics
                    )
                    
                    if it < 1000 and cosine_sim > 0.95:
                        print(f"\n⚠️ [Warning] High cosine at iter {it}: {cosine_sim:.4f}")
            # Simplified progress bar
            if step % 10 == 0 or step == len(loader) - 1:
                pbar.set_postfix(
                    L=f"{loss.item():.2e}",
                    GL=f"{loss_global.item():.2e}",
                    LocL=f"{loss_local.item():.2e}",
                    ML=f"{loss_mask.item():.2e}",
                    COS=f"{cosine_sim:.4f}",
                    LR=f"{optimizer.param_groups[0]['lr']:.2e}",
                    MOM=f"{current_momentum:.4f}"
                )
            
            it += 1

        # Save checkpoint
        should_save = ((epoch + 1) % args.save_every_epochs == 0) or ((epoch + 1) == args.max_epochs)
        
        if should_save:
            ckpt = {
                "epoch": epoch,
                "iteration": it,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            ckpt_path = output_dir / f"SSL_epoch_{epoch + 1:04d}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"[Info] Saved checkpoint {ckpt_path}")

            shutil.copy(str(ckpt_path), output_dir / "SSL_latest.pth")

            if args.keep_last_checkpoints > 0:
                ckpts = sorted(output_dir.glob("SSL_epoch_*.pth"))
                if len(ckpts) > args.keep_last_checkpoints:
                    for old_ckpt in ckpts[:-args.keep_last_checkpoints]:
                        old_ckpt.unlink()
                        print(f"[Info] Deleted old checkpoint {old_ckpt.name}")

        # Compute comprehensive metrics every 10 epochs
        if ((epoch + 1) % args.save_every_epochs == 0) or ((epoch + 1) == args.max_epochs):
            if args.visualize_train_data or args.visualize_test_data:
                compute_and_log_all_metrics(
                    args, teacher, device, epoch, logs_dir, enhanced_logger, mode="pretrain"
                )

    instant_logger.plot(logs_dir)
    print(f"[Info] SSL pretraining complete. Logs: {logs_dir}")


# -------------------------------------------------------
# ArcFace Finetuning
# -------------------------------------------------------

def finetune_arcface(args):
    """Finetune with ArcFace metric learning."""
    try:
        from pytorch_metric_learning import losses
    except ImportError:
        raise ImportError(
            "Please install pytorch-metric-learning:\n"
            "pip install pytorch-metric-learning"
        )
    
    device = resolve_device(args.device)
    set_seed(args.seed)
    
    # Load pretrained model
    resume_path = args.resume if args.resume else ""
    checkpoint_path = resume_path or args.checkpoint
    if not checkpoint_path:
        raise ValueError("finetune mode requires --checkpoint or --resume.")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Check if it's an ArcFace checkpoint (for resume) or SSL checkpoint
    if "loss_func" in ckpt:
        print(f"[Info] Resuming ArcFace training from {checkpoint_path}")
        model = SSLModel(args.model_name, args.out_dim, img_size=args.global_crop_size).to(device)
        
        embed_dim = model.backbone.num_features
        model.projector = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, args.metric_embed_dim),
        ).to(device)
        
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", -1) + 1
    else:
        print(f"[Info] Loading SSL pretrained checkpoint: {checkpoint_path}")
        model = SSLModel(args.model_name, args.out_dim, img_size=args.global_crop_size).to(device)

        # adapt models
        state_dict = ckpt["teacher"]
        source_model_name = ckpt.get("args", {}).get("model_name", "vit_small_patch16_224")
        target_model_name = args.model_name
        
        state_dict = adapt_state_dict_keys(
            state_dict, 
            source_model_name, 
            target_model_name
        )

        model.load_state_dict(ckpt["teacher"])
        
        # Freeze first 70% of backbone layers
        total_blocks = len(list(model.backbone.blocks))
        freeze_until = int(total_blocks * 0.7)
        
        for idx, block in enumerate(model.backbone.blocks):
            if idx < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
        
        print(f"[Info] Frozen first {freeze_until}/{total_blocks} transformer blocks")
        
        embed_dim = model.backbone.num_features
        model.projector = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, args.metric_embed_dim),
        ).to(device)
        
        start_epoch = 0
    
    # Dataset
    transform = build_eval_transform(args.global_crop_size)
    dataset = CSVImageDataset(args.train_data, args.input_images_dir, transform, require_label=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=resolve_num_workers(args.num_workers),
    )
    
    # Get number of classes
    if "label_to_idx" in ckpt and ckpt["label_to_idx"]:
        label_to_idx = ckpt["label_to_idx"]
        print("[Info] Loaded label_to_idx from checkpoint (resume).")
    else:
        all_labels = dataset.labels
        label_to_idx = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}
    missing = set(dataset.labels) - set(label_to_idx.keys())
    if missing:
        raise ValueError(f"Labels {missing} not found in checkpoint label_to_idx.")    
    num_classes = len(label_to_idx)
    print(f"[Info] Training ArcFace on {num_classes} classes")
    
    # ArcFace loss
    loss_func = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=args.metric_embed_dim,
        margin=28.6,
        scale=64,
    ).to(device)
    
    if "loss_func" in ckpt:
        loss_func.load_state_dict(ckpt["loss_func"])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(loss_func.parameters()),
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )
    
    if "optimizer" in ckpt and "loss_func" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    
    output_dir = Path(args.out_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize enhanced logger for ArcFace
    instant_logger = InstantMetricsLogger(logs_dir / "instant_metrics.finetune.csv", mode="finetune")
    enhanced_logger = EnhancedMetricsLogger(logs_dir / "metrics.arcface.csv", mode="arcface")
    
    # Training loop
    it = ckpt.get("iteration", start_epoch * len(loader))
    for epoch in range(start_epoch, args.finetune_epochs):
        model.train()
        loss_func.train()
    
        pbar = tqdm(loader, desc=f"Finetune Epoch {epoch+1}/{args.finetune_epochs}", ncols=100)
        total_loss = 0.0
    
        for step, (imgs, _, labels) in enumerate(pbar):
            imgs = imgs.to(device)
            labels = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long).to(device)
        
            embeddings, _ = model(imgs)
            loss = loss_func(embeddings, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )
        
            # Log instant metrics every N iterations
            log_every = max(1, args.log_every_n_steps)
            if it % log_every == 0:
                instant_metrics = compute_instant_metrics_arcface(model, imgs, labels, device)
                instant_logger.log(
                    iteration=it,
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    **instant_metrics
                )
        
            it += 1
    
        avg_loss = total_loss / len(loader)
        
        print(f"[Finetune] Epoch {epoch+1}/{args.finetune_epochs} - Avg Loss: {avg_loss:.4f}")
        
        # kNN-Acc (k=1)
        if args.visualize_train_data or args.visualize_test_data:
            model.eval()
            with torch.no_grad():
                if args.visualize_train_data:
                    transform = build_eval_transform(args.global_crop_size)
                    dataset_eval = CSVImageDataset(args.train_data, args.input_images_dir, transform, require_label=True)
                    loader_eval = DataLoader(dataset_eval, batch_size=64, shuffle=False, num_workers=resolve_num_workers(args.num_workers))
                    
                    all_embeddings = []
                    all_labels = []
                    for imgs, _, lbls in loader_eval:
                        imgs = imgs.to(device)
                        emb, _ = model(imgs)
                        all_embeddings.append(emb.cpu().numpy())
                        all_labels.extend([label_to_idx[l] for l in lbls])
                    
                    embeddings_np = np.concatenate(all_embeddings, axis=0)
                    labels_np = np.array(all_labels)
                    
                    from sklearn.preprocessing import normalize
                    embeddings_norm = normalize(embeddings_np, norm='l2')
                    from sklearn.metrics.pairwise import cosine_similarity
                    sim_matrix = cosine_similarity(embeddings_norm)
                    np.fill_diagonal(sim_matrix, -np.inf)
                    nearest_idx = sim_matrix.argmax(axis=1)
                    knn_acc = (labels_np[nearest_idx] == labels_np).mean() * 100
                    
                    print(f"         kNN-Acc(k=1): {knn_acc:.2f}%")
            
            model.train()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every_epochs == 0 or (epoch + 1) == args.finetune_epochs:
            ckpt_path = output_dir / f"arcface_epoch_{epoch+1:04d}.pth"
            torch.save({
                "epoch": epoch,
                "iteration": it,
                "model": model.state_dict(),
                "loss_func": loss_func.state_dict(),
                "optimizer": optimizer.state_dict(),
                "label_to_idx": label_to_idx,
            }, ckpt_path)
            print(f"[Info] Saved ArcFace checkpoint {ckpt_path}")
            
            shutil.copy(str(ckpt_path), output_dir / "arcface_latest.pth")
        
        # Compute comprehensive metrics every 10 epochs
        if ((epoch + 1) % args.save_every_epochs == 0) or ((epoch + 1) == args.finetune_epochs):
            if args.visualize_train_data or args.visualize_test_data:
                compute_and_log_all_metrics(
                    args, model, device, epoch, logs_dir, enhanced_logger, mode="arcface"
                )
    
    instant_logger.plot(logs_dir)
    print("[Info] ArcFace finetuning complete!")
    

# -------------------------------------------------------
# Embeddings / visualization helpers
# -------------------------------------------------------

def extract_topk_patch_embeddings(
    model, 
    images: torch.Tensor, 
    k: int, 
    device: torch.device
) -> np.ndarray:
    """
    Extract top-K patch embeddings based on L2 norm.
    
    Args:
        model: SSLModel instance
        images: [B, 3, H, W] batch of images
        k: number of top patches to select
        device: torch device
    
    Returns:
        [B, K*D] concatenated top-K patch embeddings
    """
    with torch.no_grad():
        _, all_tokens = model(images, return_all_patches=True)  # [B, N+1, D]
        patch_tokens = all_tokens[:, 1:, :]  # [B, N, D]
        patch_norms = torch.norm(patch_tokens, p=2, dim=-1)  # [B, N]
        _, topk_indices = torch.topk(patch_norms, k=k, dim=1)  # [B, K]
        B, N, D = patch_tokens.shape
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)  # [B, K, D]
        topk_patches = torch.gather(patch_tokens, dim=1, index=topk_indices_expanded)  # [B, K, D]
        topk_patches_flat = topk_patches.reshape(B, -1)  # [B, K*D]
        
    return topk_patches_flat.cpu().numpy()


def apply_pca_reduction(embeddings: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Apply PCA dimensionality reduction with dynamic dimension adjustment.
    
    Args:
        embeddings: [N, D] original embeddings
        target_dim: target dimensionality (will be adjusted if N < target_dim)
    
    Returns:
        [N, target_dim] reduced embeddings (or [N, N-1] if samples are too few)
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[Warning] scikit-learn not available, skipping PCA")
        return embeddings
    
    N, D = embeddings.shape
    
    # Skip PCA if already low-dimensional
    if D <= target_dim:
        print(f"[Info] Embedding dim {D} <= target {target_dim}, skip PCA")
        return embeddings
    
    # Adjust target_dim based on sample size
    # PCA can reduce to at most min(n_samples-1, n_features)
    max_components = min(N - 1, D)
    
    if target_dim > max_components:
        adjusted_dim = max_components
        print(f"[Warning] Requested PCA dim {target_dim} > max possible {max_components}")
        print(f"[Info] Adjusting PCA target to {adjusted_dim}D (limited by {N} samples)")
        target_dim = adjusted_dim
    
    # Apply PCA
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    pca = PCA(n_components=target_dim, random_state=42)
    embeddings_reduced = pca.fit_transform(embeddings_scaled)
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"[Info] PCA: {D}D → {target_dim}D "
          f"(explained variance: {explained_var:.2%}, {N} samples)")
    
    return embeddings_reduced


def compute_embeddings_from_csv(
    model, 
    csv_path, 
    root_dir, 
    transform, 
    device, 
    use_projector: bool,
    token_mode: str = "cls",
    topk_patches: int = 20
):
    """
    Compute embeddings from CSV with support for patch token mode.
    
    Args:
        model: trained model
        csv_path: path to CSV file
        root_dir: root directory for images
        transform: image transforms
        device: torch device
        use_projector: whether to use projector output
        token_mode: "cls", "patch-topk", or "attention-pool"
        topk_patches: number of top-K patches (10/20/30)
    
    Returns:
        (embeddings, filenames, labels)
    """
    dataset = CSVImageDataset(csv_path, root_dir, transform, require_label=False)
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=resolve_num_workers(4),
        pin_memory=(device.type == "cuda"),
        persistent_workers=False
    )
    model.eval()
    
    feats, files = [], []
    
    with torch.no_grad():
        for imgs, rels, _ in tqdm(loader, desc="Embed CSV", ncols=120, leave=False):
            imgs = imgs.to(device)
            
            if token_mode == "attention-pool":
                if hasattr(model, 'attention_pool'):
                    backbone_feats = model.backbone.forward_features(imgs)
                    tokens = backbone_feats["x"] if isinstance(backbone_feats, dict) else backbone_feats
                    patch_tokens = tokens[:, 1:, :]
                    pooled, _ = model.attention_pool(patch_tokens)
                    batch_feats = pooled.cpu().numpy()
                else:
                    raise ValueError(
                        "Model doesn't have attention_pool. "
                        "Set use_attention_pooling=True when creating model."
                    )
            
            elif token_mode == "patch-topk":
                # Extract top-K patch embeddings
                batch_feats = extract_topk_patch_embeddings(
                    model, imgs, k=topk_patches, device=device
                )
            
            else:  # cls mode
                if use_projector:
                    proj, _ = model(imgs, return_tokens=True)
                    batch_feats = proj.cpu().numpy()
                else:
                    _, cls = model(imgs)
                    batch_feats = cls.cpu().numpy()
            
            feats.append(batch_feats)
            files.extend(rels)
    
    embeddings = np.concatenate(feats, axis=0)
    
    # Apply PCA only for patch-topk mode (NOT for attention-pool)
    if token_mode == "patch-topk":
        pca_dim_map = {10: 256, 20: 512, 30: 1024}
        preferred_dim = pca_dim_map.get(topk_patches, 512)
        
        N = embeddings.shape[0]
        max_possible_dim = min(N - 1, embeddings.shape[1])
        target_dim = min(preferred_dim, max_possible_dim)
        
        if target_dim < preferred_dim:
            print(f"[Info] Limited by {N} samples, using PCA dim={target_dim} instead of {preferred_dim}")
        
        print(f"[Info] Applying PCA reduction for Top-{topk_patches} patches...")
        embeddings = apply_pca_reduction(embeddings, target_dim=target_dim)
    
    return embeddings, files, dataset.labels


def save_embeddings_csv(output_path: Path, image_list: List[str], embeddings: np.ndarray):
    cols = ["image"] + [f"E{idx:04d}" for idx in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=cols[1:])
    df.insert(0, "image", image_list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Info] Saved embeddings to {output_path} (shape={embeddings.shape})")


def run_umap_wrapper(features, labels, out_dir: Path, filename: str, title: str, args):
    if labels is None:
        print("[Warning] Labels missing -> skip UMAP.")
        return
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import umap.umap_ as umap
        from sklearn.preprocessing import Normalizer
    except ImportError as e:
        print(f"[Warning] Missing visualization dependency ({e}); skipped.")
        return

    if len(labels) < 2:
        print(f"[Warning] Need >=2 samples for UMAP ({filename}); skipped.")
        return

    feats_norm = Normalizer(norm="l2").fit_transform(features)
    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
        n_components=2,
        n_jobs=1,
        verbose=False
    )
    coords = reducer.fit_transform(feats_norm)
    df = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
    df["label"] = pd.Series(labels).astype(str)

    if args.visualize_class_number and args.visualize_class_number > 0:
        counts = df["label"].value_counts()
        keep = counts.head(args.visualize_class_number).index.tolist()
        dropped = len(counts) - len(keep)
        df = df[df["label"].isin(keep)]
        if df.empty:
            print("[Warning] No samples remain after class filter; skip UMAP.")
            return
        print(f"[Info] Showing top {len(keep)} classes, dropped {dropped} others.")

    unique_labels = df["label"].unique().tolist()
    if len(unique_labels) <= 10:
        palette = sns.color_palette("tab10", len(unique_labels))
    elif len(unique_labels) <= 20:
        palette = sns.color_palette("tab20", len(unique_labels))
    else:
        palette = sns.color_palette("husl", len(unique_labels))
    cmap = dict(zip(unique_labels, palette))

    max_labels_per_col = 30
    n_cols = max(1, math.ceil(len(unique_labels) / max_labels_per_col))
    fig_width = 10 + 1.2 * n_cols
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    for lbl in unique_labels:
        sub = df[df["label"] == lbl]
        ax.scatter(sub["UMAP1"], sub["UMAP2"],
                   s=30, alpha=0.85, label=str(lbl),
                   color=cmap[lbl], edgecolors="none")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        title="Label",
        frameon=True,
        fontsize="small",
        title_fontsize="small",
        ncol=n_cols,
        columnspacing=0.8,
        handlelength=1.0,
        borderaxespad=0.2,
    )
    legend._legend_box.align = "left"
    legend_space = min(0.32, 0.08 * n_cols)
    right_rect = 1 - legend_space
    fig.tight_layout(rect=[0, 0, right_rect, 1])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=300, format="pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"[Info] Saved UMAP plot to {out_dir / filename}")


# -------------------------------------------------------
# Extraction mode
# -------------------------------------------------------

def finetune_attention_query(
    model: nn.Module,
    train_csv: str,
    root_dir: str,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    num_epochs: int = 20,
    lr: float = 1e-3,
    global_crop_size: int = 224,
    logs_dir: Path = None,
    args = None
) -> nn.Module:
    """
    Finetune attention pooling query vector using contrastive learning.
    """
    print("\n" + "="*80)
    print("🔧 Automatic Attention Query Finetuning")
    print("="*80)
    
    # Freeze everything except attention_pool.query
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable gradients only for query
    if hasattr(model, 'attention_pool'):
        if hasattr(model.attention_pool, 'query'):
            model.attention_pool.query.requires_grad = True
            trainable_params = [model.attention_pool.query]
            print(f"[Info] Trainable parameters: {model.attention_pool.query.numel():,}")
        else:
            # For gated attention, train all attention_pool parameters
            for param in model.attention_pool.parameters():
                param.requires_grad = True
            trainable_params = list(model.attention_pool.parameters())
            total_params = sum(p.numel() for p in trainable_params)
            print(f"[Info] Trainable parameters: {total_params:,}")
    else:
        raise ValueError("Model doesn't have attention_pool module")
    
    # Dataset
    transform = build_eval_transform(global_crop_size)
    dataset = CSVImageDataset(train_csv, root_dir, transform, require_label=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    
    # Get label mapping
    all_labels = dataset.labels
    unique_labels = sorted(set(all_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    
    print(f"[Info] Training on {len(dataset)} samples, {num_classes} classes")
    print(f"[Info] Finetuning for {num_epochs} epochs with lr={lr}")
    
    # Optimizer (only for query)
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    # Temperature for contrastive loss
    temperature = 0.07
    
    if logs_dir:
        instant_logger = InstantMetricsLogger(logs_dir / "instant_metrics.attention_query.csv", mode="finetune")
    else:
        instant_logger = None
    
    # Training loop
    model.train()
    it = 0
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Query Finetune [{epoch+1}/{num_epochs}]", ncols=100)
        
        for step, (imgs, _, labels) in enumerate(pbar):
            imgs = imgs.to(device)
            
            # Convert labels to indices
            label_indices = torch.tensor(
                [label_to_idx[l] for l in labels], 
                dtype=torch.long
            ).to(device)
            
            # Forward pass through backbone (frozen)
            with torch.no_grad():
                backbone_feats = model.backbone.forward_features(imgs)
                tokens = backbone_feats["x"] if isinstance(backbone_feats, dict) else backbone_feats
                patch_tokens = tokens[:, 1:, :]  # [B, N, D]
            
            # Attention pooling (trainable query)
            pooled, attn_weights = model.attention_pool(patch_tokens)  # [B, D]
            
            # Normalize embeddings
            pooled = F.normalize(pooled, dim=-1)
            
            # Compute similarity matrix
            sim_matrix = pooled @ pooled.T / temperature  # [B, B]
            
            # Create positive mask (same label)
            pos_mask = (label_indices.unsqueeze(0) == label_indices.unsqueeze(1)).float()
            pos_mask.fill_diagonal_(0)  # Exclude self
            
            # Create negative mask (different label)
            neg_mask = 1 - pos_mask
            neg_mask.fill_diagonal_(0)
            
            # InfoNCE loss
            exp_sim = torch.exp(sim_matrix)
            pos_sim = (sim_matrix * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
            neg_sim = torch.log((exp_sim * neg_mask).sum(dim=1) + 1e-8)
            loss = -pos_sim + neg_sim
            loss = loss.mean()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy (nearest neighbor)
            with torch.no_grad():
                sim_matrix_np = sim_matrix.cpu().numpy()
                np.fill_diagonal(sim_matrix_np, -np.inf)
                nearest_idx = sim_matrix_np.argmax(axis=1)
                nearest_labels = label_indices[nearest_idx]
                correct += (nearest_labels == label_indices).sum().item()
                total += len(label_indices)
            
            # Simplified progress bar
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{lr:.2e}"
            )
            
            # Log instant metrics
            if instant_logger and it % log_every_n_steps == 0:
                instant_metrics = compute_instant_metrics_arcface(model, imgs, label_indices, device)
                instant_logger.log(
                    iteration=it,
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    lr=lr,
                    **instant_metrics
                )
            
            it += 1
        
        avg_loss = total_loss / len(loader)
        avg_acc = 100 * correct / total
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, kNN-Acc(k=1): {avg_acc:.2f}%")
    
    if instant_logger:
        instant_logger.plot(logs_dir)
    
    print("="*80)
    print("✅ Query finetuning complete!")
    print("="*80 + "\n")
    
    # Set back to eval mode
    model.eval()
    return model


def adapt_state_dict_keys(state_dict, source_model_name, target_model_name):
    """
    Adapt checkpoint keys to match target model architecture.
    
    Args:
        state_dict: checkpoint state dict
        source_model_name: model name used during training (from checkpoint)
        target_model_name: model name for current extraction
    
    Returns:
        adapted state_dict
    """
    def get_model_type(name):
        name_lower = name.lower()
        if 'eva' in name_lower:
            return 'eva'
        elif 'swin' in name_lower:
            return 'swin'
        elif 'convnext' in name_lower:
            return 'convnext'
        else:
            return 'standard'
    
    source_type = get_model_type(source_model_name)
    target_type = get_model_type(target_model_name)
    
    if source_type == target_type:
        return state_dict
    
    print(f"[Info] Adapting checkpoint: {source_type} → {target_type}")
    
    new_state_dict = {}
    
    key_mappings = {
        ('eva', 'standard'): {
            'backbone.fc_norm': 'backbone.norm',
        },
        ('standard', 'eva'): {
            'backbone.norm': 'backbone.fc_norm',
        },
    }
    
    mapping = key_mappings.get((source_type, target_type), {})
    
    for k, v in state_dict.items():
        new_key = k
        
        for old_prefix, new_prefix in mapping.items():
            if k.startswith(old_prefix):
                new_key = k.replace(old_prefix, new_prefix, 1)
                print(f"  Mapped: {k} -> {new_key}")
                break
        
        new_state_dict[new_key] = v
    
    return new_state_dict


def extract_embeddings(args):
    """
    Enhanced extraction mode with metrics computation and visualization.
    
    Behavior:
    - If --visualize_train_data or --visualize_test_data provided:
      Extract from CSV files and compute metrics (with labels if available)
    - Otherwise: Extract from entire --input_images_dir folder
    - If --token_mode is "attention-pool": automatically finetune query before extraction
    """
    device = resolve_device(args.device)
    transform = build_eval_transform(args.extract_size)
    use_attn_pool = (args.token_mode == "attention-pool")
    
    # Determine checkpoint path and finetuned checkpoint path
    checkpoint_path = Path(args.checkpoint)
    
    if use_attn_pool:
        # e.g., SSL_epoch_0050.pth -> SSL_epoch_0050.lightweight_pooling.pth
        finetuned_ckpt_path = checkpoint_path.parent / (
            checkpoint_path.stem + f".{args.attention_pooling_type}_pooling.pth"
        )
        
        if finetuned_ckpt_path.exists():
            print(f"[Info] Found existing finetuned checkpoint: {finetuned_ckpt_path}")
            print(f"[Info] Loading finetuned attention pooling model...")
            ckpt = torch.load(finetuned_ckpt_path, map_location="cpu", weights_only=False)
            use_existing_finetuned = True
        else:
            print(f"[Info] No finetuned checkpoint found at {finetuned_ckpt_path}")
            print(f"[Info] Will finetune query and save to {finetuned_ckpt_path}")
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            use_existing_finetuned = False
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        use_existing_finetuned = False
    
    # Load model
    model = SSLModel(
        args.model_name, 
        args.out_dim, 
        img_size=args.global_crop_size,
        use_attention_pooling=use_attn_pool,
        pooling_type=args.attention_pooling_type
    ).to(device)
    
    if "loss_func" in ckpt:
        # ArcFace checkpoint
        embed_dim = model.backbone.num_features
        model.projector = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, args.metric_embed_dim),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        print(f"[Info] Loaded ArcFace checkpoint: {args.checkpoint}")
    else:
        # SSL checkpoint
        source_model_name = ckpt.get("args", {}).get("model_name", "vit_small_patch16_224")
        target_model_name = args.model_name
        
        state_dict = ckpt["teacher"]
        state_dict = adapt_state_dict_keys(
            state_dict, 
            source_model_name, 
            target_model_name
        )

        if use_attn_pool:
            if use_existing_finetuned:
                # Load fully finetuned model
                model.load_state_dict(ckpt["teacher"])
                print(f"[Info] Loaded finetuned SSL checkpoint with Attention Pooling "
                      f"({args.attention_pooling_type})")
            else:
                # Load backbone and projector, initialize attention_pool randomly
                model_state = model.state_dict()
                pretrained_state = {
                    k: v for k, v in state_dict.items()
                    if k in model_state and "attention_pool" not in k
                }
                model_state.update(pretrained_state)
                model.load_state_dict(model_state)
                
                print(f"[Info] Loaded SSL Teacher checkpoint: {args.checkpoint}")
                print(f"[Info] Initialized Attention Pooling ({args.attention_pooling_type})")
                print(f"[Info] Attention pooling parameters: "
                      f"{sum(p.numel() for p in model.attention_pool.parameters()):,}")
                
                # AUTO-FINETUNE QUERY
                if not args.visualize_train_data:
                    raise ValueError(
                        "Error: --visualize_train_data is required for attention-pool mode "
                        "to finetune the query vector. Please provide training CSV."
                    )
                
                print(f"\n[Info] Starting automatic query finetuning...")
                model = finetune_attention_query(
                    model=model,
                    train_csv=args.visualize_train_data,
                    root_dir=args.input_images_dir,
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=resolve_num_workers(args.num_workers),
                    num_epochs=args.attention_pooling_epochs,
                    lr=1e-3,
                    global_crop_size=args.global_crop_size
                )
                
                # Save finetuned checkpoint
                print(f"[Info] Saving finetuned checkpoint to {finetuned_ckpt_path}")
                torch.save({
                    "teacher": model.state_dict(),
                    "args": vars(args)
                }, finetuned_ckpt_path)
                print(f"[Info] ✅ Finetuned checkpoint saved!")
        else:
            # Regular CLS mode
            model.load_state_dict(ckpt["teacher"])
            print(f"[Info] Loaded SSL Teacher checkpoint: {args.checkpoint}")
    
    model.eval()
    output_dir = Path(args.out_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine extraction mode
    has_csv_input = args.visualize_train_data or args.visualize_test_data
    
    if has_csv_input:
        # ========== CSV-based extraction with metrics ==========
        print("[Info] CSV-based extraction mode enabled")
        
        # Initialize metrics logger
        enhanced_logger = EnhancedMetricsLogger(
            logs_dir / "metrics.extract.csv", 
            mode="pretrain"
        )
        
        # Process train CSV if provided
        if args.visualize_train_data:
            print(f"\n[Extract] Processing train CSV: {args.visualize_train_data}")
            feats, files, labels = compute_embeddings_from_csv(
                model=model,
                csv_path=args.visualize_train_data,
                root_dir=args.input_images_dir,
                transform=transform,
                device=device,
                use_projector=args.use_projector_output,
                token_mode=args.token_mode,
                topk_patches=args.topk_patches,
            )
            
            # Save embeddings
            save_embeddings_csv(
                output_dir / "embeddings.train.csv", 
                files, 
                feats
            )
            
            # Compute metrics (works with or without labels)
            print("[Metrics] Computing metrics for train set...")
            feats_eval, labels_eval = maybe_subsample_for_metrics(
                feats, labels, args.metrics_sample_size
            )
            metrics = compute_all_metrics(
                feats_eval, labels_eval, mode="pretrain", compute_linear_probing=True
            )
            enhanced_logger.log(0, "train", metrics)
            
            print(f"[Metrics] Train set results:")
            for k, v in metrics.items():
                if v != "":
                    print(f"  {k}: {v:.4f}")
            
            # UMAP visualization (only if labels exist)
            if labels is not None and len(labels) > 0:
                run_umap_wrapper(
                    features=feats,
                    labels=labels,
                    out_dir=logs_dir,
                    filename="umap.train.extract.pdf",
                    title=f"UMAP Train - Extract ({args.token_mode})",
                    args=args,
                )
        
        # Process test CSV if provided
        if args.visualize_test_data:
            print(f"\n[Extract] Processing test CSV: {args.visualize_test_data}")
            feats, files, labels = compute_embeddings_from_csv(
                model=model,
                csv_path=args.visualize_test_data,
                root_dir=args.input_images_dir,
                transform=transform,
                device=device,
                use_projector=args.use_projector_output,
                token_mode=args.token_mode,
                topk_patches=args.topk_patches,
            )
            
            # Save embeddings
            save_embeddings_csv(
                output_dir / "embeddings.test.csv", 
                files, 
                feats
            )
            
            # Compute metrics (works with or without labels)
            feats_eval, labels_eval = maybe_subsample_for_metrics(
                feats, labels, args.metrics_sample_size
            )
            print("[Metrics] Computing metrics for test set...")
            metrics = compute_all_metrics(
                feats_eval, labels_eval, mode="pretrain", compute_linear_probing=True
            )
            enhanced_logger.log(0, "test", metrics)
            
            print(f"[Metrics] Test set results:")
            for k, v in metrics.items():
                if v != "":
                    print(f"  {k}: {v:.4f}")
            
            # UMAP visualization (only if labels exist)
            if labels is not None and len(labels) > 0:
                run_umap_wrapper(
                    features=feats,
                    labels=labels,
                    out_dir=logs_dir,
                    filename="umap.test.extract.pdf",
                    title=f"UMAP Test - Extract ({args.token_mode})",
                    args=args,
                )
        
        print(f"\n[Info] CSV-based extraction complete. Results saved to {output_dir}")
    
    else:
        # ========== Folder-based extraction (original behavior) ==========
        print("[Info] Folder-based extraction mode (no CSV provided)")
        
        dataset = FolderDataset(args.input_images_dir, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=resolve_num_workers(args.num_workers),
        )
        
        all_embeddings, all_paths = [], []
        
        for imgs, paths in tqdm(loader, desc="Extract embeddings", ncols=120):
            imgs = imgs.to(device)
            with torch.no_grad():
                if args.token_mode == "attention-pool":
                    if hasattr(model, 'attention_pool'):
                        backbone_feats = model.backbone.forward_features(imgs)
                        tokens = backbone_feats["x"] if isinstance(backbone_feats, dict) else backbone_feats
                        patch_tokens = tokens[:, 1:, :]
                        pooled, _ = model.attention_pool(patch_tokens)
                        feats = pooled.cpu().numpy()
                    else:
                        raise ValueError("Model doesn't have attention_pool")
                
                elif args.token_mode == "patch-topk":
                    feats = extract_topk_patch_embeddings(
                        model, imgs, k=args.topk_patches, device=device
                    )
                
                else:  # cls mode
                    if args.use_projector_output:
                        proj, _ = model(imgs, return_tokens=True)
                        feats = proj.cpu().numpy()
                    else:
                        _, cls = model(imgs)
                        feats = cls.cpu().numpy()
            
            all_embeddings.append(feats)
            all_paths.extend(paths)
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Apply PCA if in patch-topk mode
        if args.token_mode == "patch-topk":
            pca_dim_map = {10: 256, 20: 512, 30: 1024}
            preferred_dim = pca_dim_map.get(args.topk_patches, 512)
    
            # Adjust based on sample size
            N = embeddings.shape[0]
            max_possible_dim = min(N - 1, embeddings.shape[1])
            target_dim = min(preferred_dim, max_possible_dim)
    
            if target_dim < preferred_dim:
                print(f"[Info] Limited by {N} samples, using PCA dim={target_dim} instead of {preferred_dim}")
    
            print(f"[Info] Applying PCA reduction for Top-{args.topk_patches} patches...")
            embeddings = apply_pca_reduction(embeddings, target_dim=target_dim)
 
        # Save embeddings
        save_embeddings_csv(output_dir / "embeddings.csv", all_paths, embeddings)
        
        # Compute unsupervised metrics (no labels)
        print("[Metrics] Computing unsupervised metrics for entire folder...")
        embeddings_eval, _ = maybe_subsample_for_metrics(
            embeddings, None, args.metrics_sample_size
        )
        metrics = compute_all_metrics(
            embeddings_eval, labels=None, mode="pretrain", compute_linear_probing=False
        )
        
        # Save metrics log
        metrics_file = logs_dir / "metrics.extract.csv"
        with metrics_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "Silhouette_Score"])
            writer.writeheader()
            writer.writerow({
                "split": "all", 
                "Silhouette_Score": metrics.get("Silhouette_Score", "")
            })
        
        print(f"[Metrics] Folder extraction results:")
        for k, v in metrics.items():
            if v != "":
                print(f"  {k}: {v:.4f}")
        
        # Save processed image paths
        processed_images_csv = output_dir / "images.processed.csv"
        pd.DataFrame({
            "image": [str(Path(p).expanduser().resolve()) for p in all_paths]
        }).to_csv(processed_images_csv, index=False)
        print(f"[Info] Saved processed image paths to {processed_images_csv}")
        
        print(f"\n[Info] Folder-based extraction complete. Results saved to {output_dir}")


# -------------------------------------------------------
# Argument parser
# -------------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(
        description="SSL pretraining / ArcFace finetuning / embeddings extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", 
                        choices=["pretrain", "finetune", "extract"], 
                        required=True,
                        help="pretrain: SSL; finetune: ArcFace metric learning; extract: inference")
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"],
                        help="Preferred device (auto fallback).")
    parser.add_argument("--model_name", default="vit_small_patch16_224",
                        help="timm backbone.")
    parser.add_argument("--out_dim", type=int, default=256,
                        help="SSL projector output dimension.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (<=0 uses all CPUs).")
    parser.add_argument("--cpus", type=int, default=12,
                        help="Number of CPU threads for computation (PyTorch/MKL).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train_data", default="train.csv",
                        help="CSV with 'image' (and 'label' for finetune/visualization).")
    parser.add_argument("--input_images_dir", default="images",
                        help="Root directory containing all images.")
    parser.add_argument("--out_dir", default="output",
                        help="Directory for checkpoints/embeddings/logs.")

    # SSL pretraining
    parser.add_argument("--max_epochs", type=int, default=50, help="Pretraining epochs.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Base learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay.")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs.")
    parser.add_argument("--global_crop_size", type=int, default=224, help="Global crop resolution.")
    parser.add_argument("--local_crop_size", type=int, default=96, help="Local crop resolution.")
    parser.add_argument("--local_crops", type=int, default=6, help="Number of local crops.")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Masked token ratio.")
    parser.add_argument("--lambda_local", type=float, default=1.5, help="Weight of local loss.")
    parser.add_argument("--lambda_mask", type=float, default=1.0, help="Weight of mask loss.")
    parser.add_argument("--teacher_momentum", type=float, default=0.995, help="Initial EMA momentum.")
    parser.add_argument("--teacher_momentum_end", type=float, default=0.999, help="Final EMA momentum.")
    parser.add_argument("--student_temp", type=float, default=0.1, help="Student temperature.")
    parser.add_argument("--teacher_temp_start", type=float, default=0.04, help="Initial teacher temperature.")
    parser.add_argument("--teacher_temp_end", type=float, default=0.07, help="Final teacher temperature.")
    parser.add_argument("--disable_cross_view_loss", action="store_true",
                        help="Disable cross-view pairing (view1×view2).")
    parser.add_argument("--resume", default="", help="Checkpoint to resume pretraining.")
    parser.add_argument("--log_every_n_steps", type=int, default=50,
                        help="Log metrics every N iterations.")
    parser.add_argument("--save_every_epochs", type=int, default=10,
                        help="Save checkpoint every N epochs.")
    parser.add_argument("--keep_last_checkpoints", type=int, default=10,
                        help="Keep only last N checkpoints (>0).")

    # ArcFace finetuning
    parser.add_argument("--finetune_epochs", type=int, default=20,
                        help="Number of ArcFace finetuning epochs.")
    parser.add_argument("--finetune_lr", type=float, default=1e-4,
                        help="Learning rate for ArcFace finetuning.")
    parser.add_argument("--metric_embed_dim", type=int, default=256,
                        help="Embedding dimension for metric learning.")

    # Checkpoint
    parser.add_argument("--checkpoint", default="",
                        help="Manual checkpoint path (auto-detect if empty for finetune/extract).")
    
    # Extraction
    parser.add_argument("--extract_size", type=int, default=224,
                        help="Resize/Crop size for embedding extraction.")
    parser.add_argument("--use_projector_output", action="store_true",
                        help="Use projector output instead of CLS token.")
    parser.add_argument("--token_mode", 
                        choices=["cls", "patch-topk", "attention-pool"],
                        default="cls",
                        help="Token extraction mode: cls (default), patch-topk, or attention-pool")
    parser.add_argument("--topk_patches", 
                        type=int, 
                        choices=[10, 20, 30], 
                        default=20,
                        help="Number of top-K patches to select (10→256D, 20→512D, 30→1024D)")
    parser.add_argument("--attention_pooling_type",
                        choices=["lightweight", "multihead", "gated"],
                        default="lightweight",
                        help="Type of attention pooling (only used when --token_mode attention-pool)")
    parser.add_argument("--attention_pooling_epochs", type=int, default=20,
                        help="Number of epochs to finetune the attention query.")
    parser.add_argument("--metrics_sample_size", type=int, default=10000,
                        help="Maximum number of samples used when computing comprehensive metrics "
                             "(<=0 disables subsampling).")

    # Visualization
    parser.add_argument("--visualize_train_data", default="",
                        help="Draw UMAP for train data (needs label column).")
    parser.add_argument("--visualize_test_data", default="",
                        help="CSV (with labels) for UMAP visualization.")
    parser.add_argument("--visualize_class_number", type=int, default=20,
                        help="Max classes to show in UMAP (<=0 = all).")
    parser.add_argument("--umap_n_neighbors", type=int, default=15, help="UMAP n_neighbors.")
    parser.add_argument("--umap_min_dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--umap_metric", default="cosine", help="UMAP metric.")
    return parser


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def train_ssl_wrapper(args):
    """Wrapper to call train_ssl with args."""
    train_ssl(args)

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Log
    logs_dir = Path(args.out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "log.txt"
    
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger
    sys.stderr = tee_logger
    
    print(f"[Info] Logging to {log_file}")
    print(f"[Info] Command: {' '.join(sys.argv)}")
    #print(f"[Info] Arguments: {vars(args)}")

    # CPU numbers
    os.environ["OMP_NUM_THREADS"] = str(args.cpus)
    os.environ["MKL_NUM_THREADS"] = str(args.cpus)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.cpus)
    torch.set_num_threads(args.cpus)
    device = resolve_device(args.device)
    if device.type == "mps":
        print(f"[Info] Using MPS device with CPU fallback enabled")
    else:
        print(f"[Info] Using device: {device}")
    
    patch_size = 16
    if 'patch14' in args.model_name or 'eva' in args.model_name.lower():
        patch_size = 14
    elif 'patch32' in args.model_name:
        patch_size = 32

    args.global_crop_size = align_to_patch(args.global_crop_size, patch_size, 'nearest')
    args.local_crop_size = align_to_patch(args.local_crop_size, patch_size, 'nearest')
    args.extract_size = align_to_patch(args.extract_size, patch_size, 'nearest')
    
    if args.global_crop_size % patch_size != 0:
        aligned_size = (args.global_crop_size // patch_size) * patch_size
        print(f"[Warning] global_crop_size {args.global_crop_size} not divisible by patch_size {patch_size}, "
              f"auto-aligned to {aligned_size}")
        args.global_crop_size = aligned_size
    
    if args.local_crop_size % patch_size != 0:
        aligned_local = (args.local_crop_size // patch_size) * patch_size
        if aligned_local < patch_size:
            aligned_local = patch_size
        print(f"[Warning] local_crop_size {args.local_crop_size} not divisible by patch_size {patch_size}, "
              f"auto-aligned to {aligned_local}")
        args.local_crop_size = aligned_local
    
    if args.extract_size % patch_size != 0:
        aligned_extract = (args.extract_size // patch_size) * patch_size
        print(f"[Warning] extract_size {args.extract_size} not divisible by patch_size {patch_size}, "
              f"auto-aligned to {aligned_extract}")
        args.extract_size = aligned_extract
    
    if args.mode in ["finetune", "extract"] and not args.checkpoint:
        try:
            args.checkpoint = auto_find_checkpoint(Path(args.out_dir), args.mode)
        except FileNotFoundError as e:
            print(f"[Error] {e}")
            return
    if args.mode == "pretrain":
        train_ssl_wrapper(args)
    elif args.mode == "finetune":
        finetune_arcface(args)
    elif args.mode == "extract":
        extract_embeddings(args)

if __name__ == "__main__":
    main()

