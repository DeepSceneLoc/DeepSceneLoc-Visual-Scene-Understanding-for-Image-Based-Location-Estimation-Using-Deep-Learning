"""
Embedding Extraction, Visualization & Similarity Analysis
DeepSceneLoc -- Semester 2, Week 9 (Phase 3: Scene Embedding & Representation Learning)

Authors:
    Krishan Yadav  (Analysis Lead)
    Jensi Paneliya (Visualization Lead)
    Anuj Kondawar  (Pipeline Integration)

This module covers the complete Phase 3 tasks:

    Week 9  -- Embedding Extraction & Visualization (PCA, t-SNE, 3-D PCA)
    Week 10 -- Embedding Similarity Analysis (cosine similarity matrix, silhouette)

Pipeline:
    1.  Load best model checkpoint (any architecture)
    2.  Remove / bypass classification head -> get raw feature embeddings
    3.  Run all test-set images through the model
    4.  Reduce with PCA -> 2-D / 3-D visualizations
    5.  Reduce with t-SNE -> 2-D visualization
    6.  Compute pairwise cosine similarity matrix -> heatmap
    7.  Silhouette coefficient per category (separability metric)
    8.  Save all figures to results/embeddings/

Usage::

    python -m src.evaluation.embedding_analysis \\
        --model efficientnet_b0 \\
        --data data/processed \\
        --resume models/checkpoints/efficientnet/EfficientNet-B0_best.pth

    # Or import and call from a notebook / training script:
    from src.evaluation.embedding_analysis import EmbeddingAnalyzer
    analyzer = EmbeddingAnalyzer(model, device)
    embeddings, labels = analyzer.extract(test_loader)
    analyzer.run_full_analysis(embeddings, labels, output_dir="results/embeddings")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# -- Optional heavy imports (skipped gracefully in dry-run) ----
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import normalize
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["Coastal", "Forest", "Mountain", "Rural", "Urban"]
_PALETTE = [
    "#E74C3C",  # Coastal  -- red
    "#2ECC71",  # Forest   -- green
    "#8E44AD",  # Mountain -- purple
    "#E67E22",  # Rural    -- orange
    "#3498DB",  # Urban    -- blue
]


# -------------------------------------------------------------
# Embedding Extractor hook
# -------------------------------------------------------------

class EmbeddingExtractorWrapper(nn.Module):
    """
    Wraps any DeepSceneLoc model and returns the penultimate-layer
    feature vector instead of the logits.

    Supports:
      - DeepSceneLocEfficientNetAdvanced  -> features before ``self.head``
      - DeepSceneLocViTAdvanced           -> CLS token (before ``self.head``)
      - DeepSceneLocResNet50              -> features before ``backbone.fc``
      - Generic: registers a forward hook on the named layer

    Args:
        model:      The model to wrap.
        layer_name: Dotted module path to the layer whose *input* we capture
                    (e.g., ``'head'``, ``'backbone.fc'``).
                    ``None`` -> auto-detect.
    """

    def __init__(self, model: nn.Module, layer_name: Optional[str] = None):
        super().__init__()
        self.model = model
        self._embedding: Optional[torch.Tensor] = None

        if layer_name is None:
            layer_name = self._auto_detect(model)

        self._hook = self._register_hook(model, layer_name)

    @staticmethod
    def _auto_detect(model: nn.Module) -> str:
        cname = model.__class__.__name__.lower()
        if "efficientnet" in cname:
            return "head"
        if "vit" in cname:
            return "head"
        if "resnet" in cname:
            return "backbone.fc"
        # fall-back
        return "head"

    def _register_hook(self, model: nn.Module, layer_name: str):
        parts  = layer_name.split(".")
        target = model
        for part in parts:
            target = getattr(target, part)

        def _hook(module, inp, out):
            # Capture the *input* to the layer = the embedding
            self._embedding = inp[0].detach().cpu()

        return target.register_forward_hook(_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model(x)
        return self._embedding

    def remove_hook(self):
        self._hook.remove()


# -------------------------------------------------------------
# EmbeddingAnalyzer
# -------------------------------------------------------------

class EmbeddingAnalyzer:
    """
    Extract embeddings and generate all Phase 3 visualizations.

    Parameters
    ----------
    model : nn.Module
        Trained scene classifier (any architecture).
    device : str | torch.device
        Computation device.
    class_names : list[str]
        Category labels (index-ordered, default: DeepSceneLoc 5-class).
    layer_name : str | None
        Which layer's *input* to capture as embedding.  ``None`` -> auto.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        class_names: list[str] = None,
        layer_name: Optional[str] = None,
    ):
        self.device      = torch.device(device)
        self.class_names = class_names or CLASS_NAMES
        self.extractor   = EmbeddingExtractorWrapper(model, layer_name)
        self.extractor.to(self.device)
        self.extractor.eval()

    # ----------------------------------------------------------
    def extract(
        self,
        loader: DataLoader,
        max_images: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the test DataLoader through the model and collect embeddings.

        Returns
        -------
        embeddings : ndarray (N, D)
        labels     : ndarray (N,)
        """
        all_embeddings, all_labels = [], []
        n_collected = 0

        with torch.no_grad():
            for imgs, lbl in loader:
                imgs = imgs.to(self.device)
                emb  = self.extractor(imgs)     # (B, D)
                all_embeddings.append(emb.numpy())
                all_labels.append(lbl.numpy())
                n_collected += len(imgs)
                if max_images and n_collected >= max_images:
                    break

        embeddings = np.concatenate(all_embeddings, axis=0)
        labels     = np.concatenate(all_labels,     axis=0)
        print(f"  Extracted {len(embeddings):,} embeddings  dim={embeddings.shape[1]}")
        return embeddings, labels

    # ----------------------------------------------------------
    def run_full_analysis(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        output_dir: str = "results/embeddings",
        tsne_perplexity: int = 40,
        tsne_iterations: int = 1200,
        seed: int = 42,
    ) -> dict:
        """
        Run the complete Phase 3 analysis and save all figures + JSON report.

        Tasks completed:
            [OK]  PCA 2-D visualization
            [OK]  PCA 3-D visualization
            [OK]  t-SNE 2-D visualization
            [OK]  Cosine similarity matrix heatmap
            [OK]  Silhouette score per category
            [OK]  JSON report

        Returns
        -------
        report : dict   (parsed JSON report with all metrics)
        """
        if not SKLEARN_OK:
            raise ImportError(
                "scikit-learn is required for embedding analysis.\n"
                "Install with: pip install scikit-learn"
            )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        report = {}

        print("\n  [1/6] PCA 2-D ...")
        pca_2d = self._pca_2d(embeddings, labels, out, seed)
        report["pca_explained_variance_2d"] = pca_2d

        print("  [2/6] PCA 3-D ...")
        pca_3d = self._pca_3d(embeddings, labels, out, seed)
        report["pca_explained_variance_3d"] = pca_3d

        print("  [3/6] t-SNE 2-D ...")
        self._tsne_2d(embeddings, labels, out, tsne_perplexity, tsne_iterations, seed)

        print("  [4/6] Cosine similarity matrix ...")
        sim_data = self._similarity_matrix(embeddings, labels, out)
        report["category_similarity_matrix"] = sim_data

        print("  [5/6] Silhouette score ...")
        sil_data = self._silhouette(embeddings, labels)
        report["silhouette"] = sil_data

        print("  [6/6] Saving JSON report ...")
        report_path = out / "embedding_analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report -> {report_path}")

        self._print_summary(report)
        return report

    # ----------------------------------------------------------
    # Internal visualization methods
    # ----------------------------------------------------------

    def _pca_2d(self, emb, labels, out, seed) -> list:
        pca = PCA(n_components=2, random_state=seed)
        projected = pca.fit_transform(emb)
        ev = pca.explained_variance_ratio_.tolist()

        fig, ax = plt.subplots(figsize=(9, 7))
        for i, name in enumerate(self.class_names):
            mask = labels == i
            ax.scatter(
                projected[mask, 0], projected[mask, 1],
                c=_PALETTE[i], label=name, alpha=0.65, s=18, linewidths=0,
            )
        ax.set_xlabel(f"PC1 ({ev[0]:.1%} variance)", fontsize=11)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%} variance)", fontsize=11)
        ax.set_title("Scene Embeddings -- PCA 2-D Projection", fontsize=13, fontweight="bold")
        ax.legend(title="Category", fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        fig.savefig(out / "pca_2d.png", dpi=150)
        plt.close(fig)
        print(f"    Saved pca_2d.png  (PC1={ev[0]:.1%}, PC2={ev[1]:.1%})")
        return ev

    def _pca_3d(self, emb, labels, out, seed) -> list:
        pca = PCA(n_components=3, random_state=seed)
        projected = pca.fit_transform(emb)
        ev = pca.explained_variance_ratio_.tolist()

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection="3d")
        for i, name in enumerate(self.class_names):
            mask = labels == i
            ax.scatter(
                projected[mask, 0], projected[mask, 1], projected[mask, 2],
                c=_PALETTE[i], label=name, alpha=0.55, s=12,
            )
        ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=9)
        ax.set_zlabel(f"PC3 ({ev[2]:.1%})", fontsize=9)
        ax.set_title("Scene Embeddings -- PCA 3-D Projection", fontsize=12, fontweight="bold")
        ax.legend(title="Category", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "pca_3d.png", dpi=150)
        plt.close(fig)
        print(f"    Saved pca_3d.png  (cumulative={sum(ev):.1%})")
        return ev

    def _tsne_2d(self, emb, labels, out, perplexity, n_iter, seed) -> None:
        # First reduce with PCA to 50-d for speed
        n_pca = min(50, emb.shape[1], emb.shape[0] - 1)
        pca   = PCA(n_components=n_pca, random_state=seed)
        emb50 = pca.fit_transform(emb)

        tsne      = TSNE(
            n_components=2, perplexity=perplexity,
            n_iter=n_iter, random_state=seed,
            learning_rate="auto", init="pca",
        )
        projected = tsne.fit_transform(emb50)

        fig, ax = plt.subplots(figsize=(9, 7))
        for i, name in enumerate(self.class_names):
            mask = labels == i
            ax.scatter(
                projected[mask, 0], projected[mask, 1],
                c=_PALETTE[i], label=name, alpha=0.65, s=18, linewidths=0,
            )
        ax.set_title(
            f"Scene Embeddings -- t-SNE 2-D  (perplexity={perplexity})",
            fontsize=13, fontweight="bold",
        )
        ax.legend(title="Category", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(True, linewidth=0.3, alpha=0.4)
        fig.tight_layout()
        fig.savefig(out / "tsne_2d.png", dpi=150)
        plt.close(fig)
        print("    Saved tsne_2d.png")

    def _similarity_matrix(self, emb, labels, out) -> dict:
        """
        Compute mean cosine similarity between each pair of categories,
        produce a heatmap, and return the matrix as a nested dict.
        """
        normed = normalize(emb, norm="l2")
        n_cls  = len(self.class_names)
        mat    = np.zeros((n_cls, n_cls))

        for i in range(n_cls):
            for j in range(n_cls):
                vi = normed[labels == i]
                vj = normed[labels == j]
                if len(vi) == 0 or len(vj) == 0:
                    mat[i, j] = 0.0
                else:
                    # Mean pairwise cosine similarity (row-of-i x row-of-j)
                    mat[i, j] = float(np.mean(vi @ vj.T))

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(n_cls)); ax.set_xticklabels(self.class_names, rotation=30, ha="right")
        ax.set_yticks(range(n_cls)); ax.set_yticklabels(self.class_names)
        ax.set_title("Category-to-Category Cosine Similarity", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean cosine sim")

        for i in range(n_cls):
            for j in range(n_cls):
                txt = ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                              color="black" if 0.3 < mat[i, j] < 0.8 else "white")
        fig.tight_layout()
        fig.savefig(out / "similarity_matrix.png", dpi=150)
        plt.close(fig)
        print("    Saved similarity_matrix.png")

        # Convert to nested dict for JSON
        result = {}
        for i, ni in enumerate(self.class_names):
            result[ni] = {self.class_names[j]: float(mat[i, j]) for j in range(n_cls)}
        return result

    def _silhouette(self, emb, labels) -> dict:
        """
        Compute overall and per-category silhouette scores.

        High silhouette (-> 1.0) means compact, well-separated clusters.
        """
        overall   = float(silhouette_score(emb, labels, metric="cosine"))
        per_sample = silhouette_samples(emb, labels, metric="cosine")

        per_class = {}
        for i, name in enumerate(self.class_names):
            mask = labels == i
            per_class[name] = float(np.mean(per_sample[mask])) if mask.any() else 0.0

        print(f"    Overall silhouette score : {overall:.4f}")
        for name, score in per_class.items():
            print(f"      {name:<12}: {score:.4f}")

        return {"overall": overall, "per_class": per_class}

    # ----------------------------------------------------------
    def _print_summary(self, report: dict) -> None:
        print("\n" + "=" * 55)
        print("  Embedding Analysis Summary")
        print("=" * 55)
        ev2 = report.get("pca_explained_variance_2d", [0, 0])
        print(f"  PCA 2-D variance explained : {sum(ev2):.1%}")
        sil = report.get("silhouette", {})
        print(f"  Silhouette score (overall) : {sil.get('overall', 0):.4f}")
        pc = sil.get("per_class", {})
        for name, s in pc.items():
            sep = "[OK]" if s > 0.2 else "[~~]"
            print(f"    {sep} {name:<12}: {s:.4f}")
        print("=" * 55)


# -------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DeepSceneLoc Embedding Analysis")
    p.add_argument("--model",   default="efficientnet_b0",
                   choices=["resnet50", "efficientnet_b0", "vit_b16"])
    p.add_argument("--data",    default="data/processed")
    p.add_argument("--resume",  required=True,
                   help="Path to trained .pth checkpoint")
    p.add_argument("--out",     default="results/embeddings",
                   help="Output directory for visualizations")
    p.add_argument("--batch",   type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-images", type=int, default=None,
                   help="Cap number of test images (useful for quick testing)")
    p.add_argument("--perplexity", type=int, default=40)
    p.add_argument("--tsne-iter",  type=int, default=1200)
    p.add_argument("--allow-cpu",  action="store_true")
    return p.parse_args()


def _build_model_for_analysis(model_name: str, n_classes: int, pretrained: bool = False):
    """Load model skeleton (pretrained=False is fine -- weights come from checkpoint)."""
    if model_name == "resnet50":
        from src.models.model import create_model
        return create_model("resnet50", num_classes=n_classes, pretrained=pretrained)
    from src.models.model_advanced import create_advanced_model
    return create_advanced_model(model_name, num_classes=n_classes, pretrained=pretrained)


def _load_ckpt(model: nn.Module, path: str, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    key  = "model_state" if "model_state" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    print(f"  Loaded weights from {path}")


def main_cli():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("cpu" if args.allow_cpu else None))
    if device is None:
        print("ERROR: CUDA not available; use --allow-cpu for debugging.")
        sys.exit(1)

    from src.preprocessing.pipeline import create_dataloaders

    _, _, test_loader = create_dataloaders(
        data_dir=args.data,
        batch_size=args.batch,
        num_workers=args.workers,
        image_size=224,
        augment_train=False,
    )

    model = _build_model_for_analysis(args.model, n_classes=5, pretrained=False)
    _load_ckpt(model, args.resume, device)

    analyzer = EmbeddingAnalyzer(model, device=str(device))
    embeddings, labels = analyzer.extract(test_loader, max_images=args.max_images)

    analyzer.run_full_analysis(
        embeddings, labels,
        output_dir=args.out,
        tsne_perplexity=args.perplexity,
        tsne_iterations=args.tsne_iter,
    )


if __name__ == "__main__":
    main_cli()
