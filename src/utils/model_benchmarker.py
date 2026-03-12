"""
Model Comparative Benchmarker
DeepSceneLoc — Semester 2, Week 10 (Comparative Model Analysis)

Author: Anuj Kondawar (Preprocessing & Pipeline Lead)

Automated comparison of ResNet-50, EfficientNet-B0, and ViT-B/16 across:
  - Accuracy  (overall, per-class, top-k)
  - Throughput (images / second on CPU and GPU)
  - Model size (parameters, MB on disk)
  - Memory    (peak VRAM / RAM during inference)
  - Training curve convergence (from checkpoint logs)

Produces both a structured dict and a printable Markdown table.

Usage::

    from src.utils.model_benchmarker import ModelBenchmarker

    benchmarker = ModelBenchmarker(device="cuda")
    
    # Register checkpoints you want to compare
    benchmarker.register("ResNet-50",     "models/checkpoints/resnet50_best.pth")
    benchmarker.register("EfficientNet",  "models/checkpoints/efficientnet_b0_best.pth")
    benchmarker.register("ViT-B/16",      "models/checkpoints/vit_best.pth")
    
    # Run full benchmark suite
    report = benchmarker.run_full_benchmark(test_loader)
    
    # Print / save
    benchmarker.print_comparison_table(report)
    benchmarker.save_report(report)
"""

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

CATEGORY_NAMES = ["Urban", "Rural", "Coastal", "Mountain", "Forest"]


# ─────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    """All benchmark results for a single model checkpoint."""
    model_name:        str
    checkpoint_path:   str

    # Accuracy
    overall_accuracy:  float = 0.0
    per_class_acc:     Dict[str, float] = field(default_factory=dict)
    top2_accuracy:     float = 0.0
    top3_accuracy:     float = 0.0

    # Speed
    inference_ms_mean: float = 0.0   # per image, averaged over test set
    inference_ms_p95:  float = 0.0
    throughput_ips:    float = 0.0   # images per second

    # Model size
    total_params_M:    float = 0.0   # millions
    model_size_MB:     float = 0.0   # checkpoint size on disk

    # Memory
    peak_vram_MB:      float = 0.0
    peak_ram_MB:       float = 0.0

    # Training history (from log file if present)
    best_val_acc:      float = 0.0
    best_epoch:        int   = 0
    final_train_loss:  float = 0.0
    final_val_loss:    float = 0.0

    # Overfitting
    overfitting_gap:   float = 0.0   # train_acc - val_acc at best epoch

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Aggregated comparison across all registered models."""
    timestamp:  str
    device:     str
    models:     List[ModelMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "device":    self.device,
            "models":    [m.to_dict() for m in self.models],
        }

    @property
    def best_accuracy_model(self) -> Optional[ModelMetrics]:
        if not self.models:
            return None
        return max(self.models, key=lambda m: m.overall_accuracy)

    @property
    def fastest_model(self) -> Optional[ModelMetrics]:
        if not self.models:
            return None
        return max(self.models, key=lambda m: m.throughput_ips)

    @property
    def smallest_model(self) -> Optional[ModelMetrics]:
        if not self.models:
            return None
        return min(
            [m for m in self.models if m.total_params_M > 0],
            key=lambda m: m.total_params_M,
            default=None,
        )


# ─────────────────────────────────────────────────────────────
# Core benchmarker
# ─────────────────────────────────────────────────────────────

class ModelBenchmarker:
    """
    Loads and compares multiple trained checkpoints on the same test set.

    Registered models are lazily loaded at benchmark time (not at registration),
    so the object is cheap to construct.
    """

    def __init__(
        self,
        device: str = "auto",
        num_classes: int = 5,
        output_dir: str = "results/metrics",
        architecture_map: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            device: ``"auto"`` (GPU if available), ``"cpu"``, ``"cuda"``.
            num_classes: Number of output classes (5 for this project).
            output_dir: Directory for JSON report files.
            architecture_map: Optional ``{model_name: arch_key}`` mapping.
                Supported arch keys: ``"resnet50"``, ``"efficientnet_b0"``, ``"vit"``.
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_classes    = num_classes
        self.output_dir     = Path(output_dir)
        self.arch_map       = architecture_map or {}
        self._registry: List[Tuple[str, str]] = []   # (name, checkpoint_path)

    # ──────────────────────────────────────────────────────────
    # Registration
    # ──────────────────────────────────────────────────────────

    def register(self, name: str, checkpoint_path: str, architecture: str = "auto"):
        """
        Add a model to the comparison set.

        Args:
            name: Display name, e.g. ``"EfficientNet-B0"``.
            checkpoint_path: Path to the ``.pth`` checkpoint.
            architecture: ``"resnet50"``, ``"efficientnet_b0"``, ``"vit"``, or
                          ``"auto"`` to infer from the checkpoint filename.
        """
        if architecture != "auto":
            self.arch_map[name] = architecture
        self._registry.append((name, checkpoint_path))
        logger.debug("Registered model '%s' from %s", name, checkpoint_path)

    # ──────────────────────────────────────────────────────────
    # Full benchmark suite
    # ──────────────────────────────────────────────────────────

    def run_full_benchmark(
        self,
        test_loader: Optional[DataLoader] = None,
        n_warmup_batches: int = 2,
    ) -> BenchmarkReport:
        """
        Run accuracy + speed + memory benchmarks for all registered models.

        Args:
            test_loader: PyTorch DataLoader for the test split. If ``None``,
                         only model-size metrics are computed.
            n_warmup_batches: GPU warm-up passes before timing.

        Returns:
            Populated ``BenchmarkReport``.
        """
        from datetime import datetime

        report = BenchmarkReport(
            timestamp = datetime.now().isoformat(timespec="seconds"),
            device    = str(self.device),
        )

        for name, ckpt_path in self._registry:
            logger.info("Benchmarking '%s' …", name)
            try:
                metrics = self._benchmark_one(name, ckpt_path, test_loader, n_warmup_batches)
                report.models.append(metrics)
            except Exception as exc:
                logger.warning("Failed to benchmark '%s': %s", name, exc)

        return report

    # ──────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────

    def print_comparison_table(self, report: BenchmarkReport):
        """Print a Markdown-formatted comparison table to stdout."""
        if not report.models:
            print("No models benchmarked.")
            return

        header = (
            "| Model | Acc (%) | Top-2 (%) | Top-3 (%) | "
            "ms/img | imgs/s | Params (M) | Size (MB) | VRAM (MB) |"
        )
        sep = "|" + "|".join(["-" * len(c) for c in header.split("|")[1:-1]]) + "|"

        print(f"\n{'='*100}")
        print("Model Comparison Report")
        print(f"Device: {report.device}  |  Timestamp: {report.timestamp}")
        print(f"{'='*100}")
        print(header)
        print(sep)

        for m in report.models:
            row = (
                f"| {m.model_name:<20} "
                f"| {m.overall_accuracy*100:>7.2f} "
                f"| {m.top2_accuracy*100:>9.2f} "
                f"| {m.top3_accuracy*100:>9.2f} "
                f"| {m.inference_ms_mean:>6.1f} "
                f"| {m.throughput_ips:>6.1f} "
                f"| {m.total_params_M:>10.2f} "
                f"| {m.model_size_MB:>9.1f} "
                f"| {m.peak_vram_MB:>9.1f} |"
            )
            print(row)

        print(f"\n  Best accuracy : {report.best_accuracy_model.model_name if report.best_accuracy_model else 'N/A'}")
        print(f"  Fastest       : {report.fastest_model.model_name if report.fastest_model else 'N/A'}")
        print(f"  Smallest      : {report.smallest_model.model_name if report.smallest_model else 'N/A'}")
        print(f"{'='*100}")

        # Per-class table
        if any(m.per_class_acc for m in report.models):
            print(f"\nPer-Class Accuracy (%):")
            cls_header = "| Class     | " + " | ".join(f"{m.model_name[:15]:<15}" for m in report.models) + " |"
            print(cls_header)
            print("|" + "|".join(["-"*len(c) for c in cls_header.split("|")[1:-1]]) + "|")
            for cat in CATEGORY_NAMES:
                row = f"| {cat:<9} |"
                for m in report.models:
                    acc = m.per_class_acc.get(cat, float("nan"))
                    row += f" {acc*100:>14.1f}% |"
                print(row)

    def save_report(
        self,
        report: BenchmarkReport,
        path: str = "results/metrics/model_comparison.json",
    ):
        """Serialize the full report to JSON."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Benchmark report saved → {dest}")

    def save_markdown_report(
        self,
        report: BenchmarkReport,
        path: str = "results/metrics/model_comparison.md",
    ):
        """Write results as a Markdown document."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Model Comparison Report",
            f"**Generated:** {report.timestamp}  ",
            f"**Device:** {report.device}  ",
            "",
            "## Summary",
            "",
            "| Model | Overall Acc | Top-2 Acc | Top-3 Acc | ms/img | imgs/s | Params (M) | Size (MB) |",
            "|-------|-------------|-----------|-----------|--------|--------|------------|-----------|",
        ]
        for m in report.models:
            lines.append(
                f"| {m.model_name} "
                f"| {m.overall_accuracy*100:.2f}% "
                f"| {m.top2_accuracy*100:.2f}% "
                f"| {m.top3_accuracy*100:.2f}% "
                f"| {m.inference_ms_mean:.1f} "
                f"| {m.throughput_ips:.1f} "
                f"| {m.total_params_M:.2f} "
                f"| {m.model_size_MB:.1f} |"
            )

        if any(m.per_class_acc for m in report.models):
            lines += ["", "## Per-Class Accuracy", ""]
            hdr = "| Class | " + " | ".join(m.model_name for m in report.models) + " |"
            lines.append(hdr)
            lines.append("|-------|" + "|".join(["---"]*len(report.models)) + "|")
            for cat in CATEGORY_NAMES:
                row = f"| {cat} |"
                for m in report.models:
                    acc = m.per_class_acc.get(cat, float("nan"))
                    row += f" {acc*100:.1f}% |"
                lines.append(row)

        if report.best_accuracy_model:
            lines += [
                "",
                "## Recommendations",
                "",
                f"- **Highest accuracy:** {report.best_accuracy_model.model_name} "
                f"({report.best_accuracy_model.overall_accuracy*100:.1f}%)",
                f"- **Fastest inference:** {report.fastest_model.model_name} "
                f"({report.fastest_model.throughput_ips:.0f} imgs/s)" if report.fastest_model else "",
                f"- **Smallest footprint:** {report.smallest_model.model_name} "
                f"({report.smallest_model.total_params_M:.1f}M params)" if report.smallest_model else "",
            ]

        with open(dest, "w") as f:
            f.write("\n".join(lines))
        print(f"Markdown report saved → {dest}")

    # ──────────────────────────────────────────────────────────
    # Internal: benchmark one model
    # ──────────────────────────────────────────────────────────

    def _benchmark_one(
        self,
        name: str,
        ckpt_path: str,
        test_loader: Optional[DataLoader],
        n_warmup: int,
    ) -> ModelMetrics:
        metrics = ModelMetrics(model_name=name, checkpoint_path=ckpt_path)

        ckpt_file = Path(ckpt_path)

        # Disk size
        if ckpt_file.exists():
            metrics.model_size_MB = ckpt_file.stat().st_size / 1e6
        else:
            logger.warning("Checkpoint not found: %s — size metrics skipped", ckpt_path)
            # Still return partial metrics rather than raising
            return metrics

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        arch = self.arch_map.get(name) or self._infer_arch(ckpt_path)
        model = self._build_model(arch, ckpt)
        model.to(self.device).eval()

        # Parameter count
        metrics.total_params_M = sum(p.numel() for p in model.parameters()) / 1e6

        # Training history from checkpoint
        if isinstance(ckpt, dict):
            metrics.best_val_acc    = float(ckpt.get("val_acc",   0.0))
            metrics.best_epoch      = int(ckpt.get("epoch",        0))
            metrics.final_val_loss  = float(ckpt.get("val_loss",   0.0))
            metrics.final_train_loss = float(ckpt.get("train_loss", 0.0))

        if test_loader is None:
            return metrics

        # Warm-up
        try:
            warm_iter = iter(test_loader)
            for _ in range(n_warmup):
                imgs, _ = next(warm_iter)
                with torch.no_grad():
                    _ = model(imgs.to(self.device))
        except StopIteration:
            pass

        # Evaluation pass
        correct_top1 = correct_top2 = correct_top3 = total = 0
        per_class_correct: Dict[int, int] = {i: 0 for i in range(self.num_classes)}
        per_class_total:   Dict[int, int] = {i: 0 for i in range(self.num_classes)}
        latencies: List[float] = []
        peak_vram_start = self._current_vram()

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)
                n      = labels.size(0)

                t0 = time.perf_counter()
                logits = model(imgs)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000 / n)

                total += n

                # Top-k accuracy
                _, topk = logits.topk(min(3, self.num_classes), dim=1)
                targets = labels.view(-1, 1)
                correct_top1 += (topk[:, :1] == targets).sum().item()
                correct_top2 += (topk[:, :2] == targets.expand_as(topk[:, :2])).any(dim=1).sum().item()
                correct_top3 += (topk[:, :3] == targets.expand_as(topk[:, :3])).any(dim=1).sum().item()

                # Per-class
                preds = topk[:, 0]
                for gt, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                    per_class_total[gt]   += 1
                    per_class_correct[gt] += int(pred == gt)

        metrics.overall_accuracy  = correct_top1 / max(total, 1)
        metrics.top2_accuracy     = correct_top2 / max(total, 1)
        metrics.top3_accuracy     = correct_top3 / max(total, 1)
        metrics.per_class_acc     = {
            CATEGORY_NAMES[i]: per_class_correct[i] / max(per_class_total[i], 1)
            for i in range(self.num_classes)
            if i < len(CATEGORY_NAMES)
        }

        if latencies:
            sorted_lat = sorted(latencies)
            metrics.inference_ms_mean = statistics.mean(latencies)
            metrics.inference_ms_p95  = sorted_lat[int(0.95 * len(sorted_lat))]
            metrics.throughput_ips    = 1000.0 / max(metrics.inference_ms_mean, 1e-9)

        metrics.peak_vram_MB = max(0.0, self._current_vram() - peak_vram_start)
        metrics.peak_ram_MB  = self._current_ram()

        return metrics

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _infer_arch(path: str) -> str:
        p = path.lower()
        if "resnet" in p:
            return "resnet50"
        if "efficientnet" in p or "efficient" in p:
            return "efficientnet_b0"
        if "vit" in p or "transformer" in p:
            return "vit"
        return "resnet50"   # safe default

    def _build_model(self, arch: str, ckpt) -> nn.Module:
        """Load model architecture and inject checkpoint weights."""
        from src.models.model import build_model   # project-local import

        model = build_model(architecture=arch, num_classes=self.num_classes)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        return model

    @staticmethod
    def _current_vram() -> float:
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1e6
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _current_ram() -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1e6
        except ImportError:
            return 0.0


# ─────────────────────────────────────────────────────────────
# Convenience: quick accuracy check from a checkpoint only
# ─────────────────────────────────────────────────────────────

def quick_model_stats(checkpoint_path: str, architecture: str = "auto") -> Dict:
    """
    Return model size and parameter count without needing a DataLoader.

    Useful for a fast sanity check before a full benchmark run.

    Args:
        checkpoint_path: Path to ``.pth`` checkpoint.
        architecture: ``"resnet50"``, ``"efficientnet_b0"``, ``"vit"``, or
                      ``"auto"`` (inferred from filename).

    Returns:
        Dict with ``total_params_M``, ``model_size_MB``, ``best_val_acc``,
        ``architecture``.
    """
    if architecture == "auto":
        architecture = ModelBenchmarker._infer_arch(checkpoint_path)

    ckpt_file = Path(checkpoint_path)
    size_mb   = ckpt_file.stat().st_size / 1e6 if ckpt_file.exists() else 0.0

    try:
        from src.models.model import build_model
        model   = build_model(architecture=architecture, num_classes=5)
        params  = sum(p.numel() for p in model.parameters()) / 1e6
    except Exception:
        params = 0.0

    best_val_acc = 0.0
    if ckpt_file.exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict):
                best_val_acc = float(ckpt.get("val_acc", 0.0))
        except Exception:
            pass

    return {
        "architecture":   architecture,
        "checkpoint_path": checkpoint_path,
        "total_params_M": round(params, 2),
        "model_size_MB":  round(size_mb, 1),
        "best_val_acc":   round(best_val_acc, 4),
    }


# ─────────────────────────────────────────────────────────────
# Smoke-test (no checkpoint files required)
# ─────────────────────────────────────────────────────────────

def _smoke_test():
    print("Smoke-testing model_benchmarker.py …")

    bench = ModelBenchmarker(device="cpu")
    bench.register("ResNet-50",    "models/checkpoints/resnet50_best.pth")
    bench.register("EfficientNet", "models/checkpoints/efficientnet_b0_best.pth")
    bench.register("ViT",          "models/checkpoints/vit_best.pth")

    # Without a test loader and without existing checkpoints we just verify
    # that the object constructs and runs without exceptions
    from datetime import datetime

    report = BenchmarkReport(
        timestamp = datetime.now().isoformat(timespec="seconds"),
        device    = "cpu",
        models    = [
            ModelMetrics(
                model_name="ResNet-50",
                checkpoint_path="mock",
                overall_accuracy=0.843,
                top2_accuracy=0.921,
                top3_accuracy=0.967,
                inference_ms_mean=185.4,
                throughput_ips=5.4,
                total_params_M=25.6,
                model_size_MB=102.1,
                per_class_acc={
                    "Urban": 0.89, "Rural": 0.82, "Coastal": 0.86,
                    "Mountain": 0.81, "Forest": 0.84,
                },
            ),
            ModelMetrics(
                model_name="EfficientNet-B0",
                checkpoint_path="mock",
                overall_accuracy=0.871,
                top2_accuracy=0.943,
                top3_accuracy=0.978,
                inference_ms_mean=118.2,
                throughput_ips=8.5,
                total_params_M=5.3,
                model_size_MB=21.4,
                per_class_acc={
                    "Urban": 0.91, "Rural": 0.85, "Coastal": 0.88,
                    "Mountain": 0.84, "Forest": 0.87,
                },
            ),
            ModelMetrics(
                model_name="ViT-B/16",
                checkpoint_path="mock",
                overall_accuracy=0.884,
                top2_accuracy=0.955,
                top3_accuracy=0.983,
                inference_ms_mean=312.7,
                throughput_ips=3.2,
                total_params_M=86.4,
                model_size_MB=330.2,
                per_class_acc={
                    "Urban": 0.93, "Rural": 0.87, "Coastal": 0.89,
                    "Mountain": 0.86, "Forest": 0.89,
                },
            ),
        ],
    )

    bench.print_comparison_table(report)

    assert report.best_accuracy_model.model_name == "ViT-B/16"
    assert report.fastest_model.model_name == "EfficientNet-B0"
    assert report.smallest_model.model_name == "EfficientNet-B0"

    print("  All checks passed.")


if __name__ == "__main__":
    _smoke_test()
