"""
Two-Stage Pipeline Optimizer
DeepSceneLoc — Semester 2, Weeks 11–13

Author: Anuj Kondawar (Preprocessing & Pipeline Lead)

This module optimises the two-stage hybrid inference pipeline:
  Stage 1 — Scene classification (PyTorch model, local)
  Stage 2 — Exact location detection (Gemini Vision API, remote)

Responsibilities (per TEAM_ALLOCATION.md):
  - Two-stage pipeline optimisation
  - Latency profiling (Stage 1 + Stage 2)
  - Error handling and fallback mechanisms
  - System reliability improvements
"""

import time
import json
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import torch
import torch.nn as nn
from PIL import Image


# ─────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────

@dataclass
class Stage1Result:
    """Output of the scene-classification stage."""
    predicted_class: str
    confidence: float
    top3: List[Tuple[str, float]]     # [(class, prob), ...]
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class Stage2Result:
    """Output of the Gemini exact-location stage."""
    exact_location: str
    country: str
    city: str
    region: str
    latitude: Optional[float]
    longitude: Optional[float]
    confidence: str                   # 'high' | 'medium' | 'low'
    landmarks: List[str]
    description: str
    latency_ms: float
    success: bool
    source: str = "gemini"            # 'gemini' | 'fallback'
    error: Optional[str] = None


@dataclass
class HybridResult:
    """Combined output of both pipeline stages."""
    stage1: Stage1Result
    stage2: Optional[Stage2Result]
    total_latency_ms: float
    pipeline_success: bool

    def to_dict(self) -> Dict:
        return {
            "stage1": asdict(self.stage1),
            "stage2": asdict(self.stage2) if self.stage2 else None,
            "total_latency_ms": self.total_latency_ms,
            "pipeline_success": self.pipeline_success,
        }


# ─────────────────────────────────────────────────────────────
# Pipeline configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Run-time configuration for the two-stage pipeline."""

    # Stage 1 thresholds
    stage1_confidence_threshold: float = 0.60
    """Minimum Stage 1 confidence required to proceed to Stage 2."""

    always_call_stage2: bool = False
    """If True, always call Gemini regardless of Stage 1 confidence."""

    # Stage 2 behaviour
    stage2_timeout_s: float = 15.0
    """Maximum seconds to wait for Gemini response."""

    stage2_max_retries: int = 2
    """Number of Gemini retry attempts on transient errors."""

    stage2_retry_delay_s: float = 1.0
    """Seconds to wait between Gemini retries."""

    # Fallback
    fallback_on_low_confidence: bool = True
    """Return a graceful fallback Stage2Result if Gemini fails."""

    # Profiling
    enable_profiling: bool = True
    """Record per-stage latency data."""

    # Checkpoint
    model_path: str = "models/checkpoints/resnet50_best.pth"
    device: str = "auto"              # 'auto' | 'cuda' | 'cpu'


# ─────────────────────────────────────────────────────────────
# Pipeline Optimizer
# ─────────────────────────────────────────────────────────────

class TwoStagePipelineOptimizer:
    """
    Orchestrates and optimises the two-stage location estimation pipeline.

    Stage 1: Local PyTorch model classifies the scene category.
    Stage 2: Gemini API identifies the exact location.

    The optimizer adds:
      - Confidence gating (skip Stage 2 for very low-confidence scenes)
      - Retry logic with exponential back-off for Gemini
      - Graceful fallback when Gemini is unavailable
      - Per-run latency profiling
    """

    CLASS_NAMES = ["Coastal", "Forest", "Mountain", "Rural", "Urban"]

    def __init__(
        self,
        model: nn.Module,
        gemini_analyzer,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Args:
            model: Trained PyTorch classification model.
            gemini_analyzer: ``GeminiLocationAnalyzer`` instance (or mock).
            config: Pipeline behaviour configuration.
        """
        self.config = config or PipelineConfig()
        self.gemini = gemini_analyzer

        # Resolve device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        self.model = model.to(self.device)
        self.model.eval()

        # Preprocessing (reuse project transforms)
        self._transform = self._build_transform()

        # Profiling history
        self._profile_records: List[Dict] = []

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def run(self, image: Image.Image) -> HybridResult:
        """
        Execute the full two-stage pipeline on a single image.

        Args:
            image: PIL RGB image.

        Returns:
            :class:`HybridResult` with both stage outputs.
        """
        t_start = time.perf_counter()

        # ── Stage 1 ──────────────────────────────────────────
        stage1 = self._run_stage1(image)

        # ── Stage 2 gate ─────────────────────────────────────
        stage2: Optional[Stage2Result] = None

        if stage1.success:
            should_call = (
                self.config.always_call_stage2
                or stage1.confidence >= self.config.stage1_confidence_threshold
            )
            if should_call:
                stage2 = self._run_stage2_with_retry(
                    image,
                    stage1.predicted_class,
                    stage1.confidence,
                )
            else:
                stage2 = self._low_confidence_fallback(stage1)

        total_ms = (time.perf_counter() - t_start) * 1000

        result = HybridResult(
            stage1=stage1,
            stage2=stage2,
            total_latency_ms=round(total_ms, 2),
            pipeline_success=stage1.success and (stage2 is None or stage2.success),
        )

        if self.config.enable_profiling:
            self._record_profile(result)

        return result

    # ──────────────────────────────────────────────────────────
    # Stage implementations
    # ──────────────────────────────────────────────────────────

    def _run_stage1(self, image: Image.Image) -> Stage1Result:
        """Run the local scene-classification model."""
        t0 = time.perf_counter()
        try:
            tensor = self._transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)[0]

            prob_values = probs.cpu().tolist()
            top_idx     = sorted(range(len(prob_values)), key=lambda i: -prob_values[i])
            top3        = [(self.CLASS_NAMES[i], round(prob_values[i], 4)) for i in top_idx[:3]]

            latency_ms = (time.perf_counter() - t0) * 1000
            return Stage1Result(
                predicted_class=top3[0][0],
                confidence=top3[0][1],
                top3=top3,
                latency_ms=round(latency_ms, 2),
                success=True,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return Stage1Result(
                predicted_class="Unknown",
                confidence=0.0,
                top3=[],
                latency_ms=round(latency_ms, 2),
                success=False,
                error=str(exc),
            )

    def _run_stage2_with_retry(
        self,
        image: Image.Image,
        category: str,
        confidence: float,
    ) -> Stage2Result:
        """Call Gemini with retry logic and exponential back-off."""
        last_error = ""
        for attempt in range(1, self.config.stage2_max_retries + 2):
            try:
                return self._run_stage2(image, category, confidence)
            except Exception as exc:
                last_error = str(exc)
                if attempt <= self.config.stage2_max_retries:
                    delay = self.config.stage2_retry_delay_s * (2 ** (attempt - 1))
                    time.sleep(delay)

        # All retries exhausted — return fallback or error result
        if self.config.fallback_on_low_confidence:
            return self._api_error_fallback(category, last_error)

        return Stage2Result(
            exact_location="Could not determine",
            country="Unknown", city="Unknown",
            region="Unknown",
            latitude=None, longitude=None,
            confidence="none",
            landmarks=[], description=last_error,
            latency_ms=0.0, success=False,
            error=last_error,
        )

    def _run_stage2(
        self,
        image: Image.Image,
        category: str,
        confidence: float,
    ) -> Stage2Result:
        """Single Gemini API call (no retry logic)."""
        t0 = time.perf_counter()
        raw = self.gemini.analyze_location(image, category, confidence)
        latency_ms = (time.perf_counter() - t0) * 1000

        return Stage2Result(
            exact_location=raw.get("exact_location", "Unknown"),
            country=raw.get("country", "Unknown"),
            city=raw.get("city", "Unknown"),
            region=raw.get("region", "Unknown"),
            latitude=raw.get("latitude"),
            longitude=raw.get("longitude"),
            confidence=raw.get("confidence", "low"),
            landmarks=raw.get("landmarks", []),
            description=raw.get("description", ""),
            latency_ms=round(latency_ms, 2),
            success="error" not in raw,
            error=raw.get("error"),
        )

    # ──────────────────────────────────────────────────────────
    # Fallbacks
    # ──────────────────────────────────────────────────────────

    def _low_confidence_fallback(self, stage1: Stage1Result) -> Stage2Result:
        """Graceful result when Stage 1 confidence is below threshold."""
        return Stage2Result(
            exact_location=f"Scene appears to be {stage1.predicted_class} — "
                           f"exact location not determined (low confidence: {stage1.confidence:.2f})",
            country="Unknown", city="Unknown",
            region="Unknown",
            latitude=None, longitude=None,
            confidence="low",
            landmarks=[],
            description=(
                f"Scene classification confidence ({stage1.confidence:.2f}) is below the "
                f"threshold ({self.config.stage1_confidence_threshold}). "
                "Gemini API was not called to conserve API quota."
            ),
            latency_ms=0.0,
            success=True,
            source="fallback",
        )

    def _api_error_fallback(self, category: str, error: str) -> Stage2Result:
        """Graceful result when Gemini API fails after all retries."""
        return Stage2Result(
            exact_location=f"Scene type: {category} (exact location unavailable)",
            country="Unknown", city="Unknown",
            region="Unknown",
            latitude=None, longitude=None,
            confidence="low",
            landmarks=[],
            description=f"Gemini API error: {error}",
            latency_ms=0.0,
            success=False,
            source="fallback",
            error=error,
        )

    # ──────────────────────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────────────────────

    def _build_transform(self):
        """Return the standard eval-time transform."""
        try:
            from src.preprocessing.transforms import get_test_transforms
            return get_test_transforms()
        except ImportError:
            import torchvision.transforms as T
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    # ──────────────────────────────────────────────────────────
    # Profiling
    # ──────────────────────────────────────────────────────────

    def _record_profile(self, result: HybridResult):
        self._profile_records.append({
            "stage1_ms":    result.stage1.latency_ms,
            "stage2_ms":    result.stage2.latency_ms if result.stage2 else 0.0,
            "total_ms":     result.total_latency_ms,
            "stage1_class": result.stage1.predicted_class,
            "stage1_conf":  result.stage1.confidence,
            "success":      result.pipeline_success,
        })

    def get_latency_report(self) -> Dict:
        """
        Return a summary of latency statistics across all profiled runs.

        Covers Stage 1, Stage 2, and total end-to-end latency.
        """
        if not self._profile_records:
            return {"error": "No profiling data collected yet."}

        def _stats(values: List[float]) -> Dict:
            return {
                "min_ms":    round(min(values), 2),
                "max_ms":    round(max(values), 2),
                "mean_ms":   round(statistics.mean(values), 2),
                "median_ms": round(statistics.median(values), 2),
                "stdev_ms":  round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
                "p95_ms":    round(sorted(values)[int(len(values) * 0.95)], 2),
            }

        stage1_latencies = [r["stage1_ms"] for r in self._profile_records]
        stage2_latencies = [r["stage2_ms"] for r in self._profile_records if r["stage2_ms"] > 0]
        total_latencies  = [r["total_ms"]  for r in self._profile_records]

        return {
            "num_samples":  len(self._profile_records),
            "stage1":       _stats(stage1_latencies),
            "stage2":       _stats(stage2_latencies) if stage2_latencies else "No Stage 2 calls",
            "total":        _stats(total_latencies),
            "success_rate": sum(r["success"] for r in self._profile_records) / len(self._profile_records),
        }

    def print_latency_report(self):
        """Print a human-readable latency report to stdout."""
        report = self.get_latency_report()
        if "error" in report:
            print(report["error"])
            return

        print(f"\n{'='*60}")
        print("Pipeline Latency Report")
        print(f"{'='*60}")
        print(f"  Samples profiled : {report['num_samples']}")
        print(f"  Success rate     : {report['success_rate']:.1%}")
        for stage_name in ("stage1", "stage2", "total"):
            s = report[stage_name]
            if isinstance(s, str):
                print(f"\n  Stage 2         : {s}")
                continue
            label = {"stage1": "Stage 1 (local)", "stage2": "Stage 2 (Gemini)", "total": "End-to-end"}.get(stage_name, stage_name)
            print(f"\n  {label}:")
            print(f"    Mean  : {s['mean_ms']} ms")
            print(f"    Median: {s['median_ms']} ms")
            print(f"    P95   : {s['p95_ms']} ms")
            print(f"    Min   : {s['min_ms']} ms   Max: {s['max_ms']} ms")
        print(f"{'='*60}")

    def save_profile(self, path: str = "results/metrics/latency_profile.json"):
        """Persist the full profiling record to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "summary": self.get_latency_report(),
                "records": self._profile_records,
            }, f, indent=2)
        print(f"Latency profile saved to {out}")


# ─────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────

def _smoke_test():
    """Verify result dataclasses and config instantiation."""
    print("Smoke-testing pipeline_optimizer.py...")

    cfg = PipelineConfig()
    assert cfg.stage1_confidence_threshold == 0.60
    assert cfg.stage2_max_retries == 2

    r = Stage1Result(
        predicted_class="Coastal", confidence=0.85,
        top3=[("Coastal", 0.85), ("Urban", 0.10), ("Forest", 0.05)],
        latency_ms=12.3, success=True,
    )
    assert r.predicted_class == "Coastal"

    r2 = Stage2Result(
        exact_location="Bondi Beach, Sydney",
        country="Australia", city="Sydney",
        region="New South Wales",
        latitude=-33.8908, longitude=151.2743,
        confidence="high",
        landmarks=["Bondi Beach"],
        description="Iconic beach in eastern Sydney.",
        latency_ms=980.5, success=True,
    )

    hr = HybridResult(stage1=r, stage2=r2, total_latency_ms=992.8, pipeline_success=True)
    d  = hr.to_dict()
    assert d["total_latency_ms"] == 992.8

    print("  All checks passed.")


if __name__ == "__main__":
    _smoke_test()
