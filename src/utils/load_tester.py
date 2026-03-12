"""
System Load Tester & Reliability Monitor
DeepSceneLoc — Semester 2, Weeks 12–13

Author: Anuj Kondawar (Preprocessing & Pipeline Lead)

Provides:
  - Throughput benchmarking (images / second)
  - Concurrent request simulation (thread-pool model)
  - Sustained stress tests (N seconds of continuous load)
  - Latency percentile breakdown per test run
  - System resource monitoring (CPU / RAM / GPU) during tests
  - HTML / JSON result reports
"""

import concurrent.futures
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    success:      bool
    latency_ms:   float
    error_msg:    str = ""


@dataclass
class LoadTestReport:
    test_name:        str
    n_requests:       int
    n_workers:        int
    duration_s:       float
    successful:       int
    failed:           int
    success_rate:     float
    throughput_rps:   float          # requests per second
    latency_mean_ms:  float
    latency_p50_ms:   float
    latency_p95_ms:   float
    latency_p99_ms:   float
    latency_max_ms:   float
    latency_min_ms:   float
    resource_samples: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Load Test: {self.test_name}")
        print(f"{'='*60}")
        print(f"  Requests      : {self.n_requests}  ({self.n_workers} workers)")
        print(f"  Duration      : {self.duration_s:.2f}s")
        print(f"  Succeeded     : {self.successful}  ({self.success_rate:.1%})")
        print(f"  Failed        : {self.failed}")
        print(f"  Throughput    : {self.throughput_rps:.2f} req/s")
        print(f"  Latency mean  : {self.latency_mean_ms:.1f} ms")
        print(f"  Latency p50   : {self.latency_p50_ms:.1f} ms")
        print(f"  Latency p95   : {self.latency_p95_ms:.1f} ms")
        print(f"  Latency p99   : {self.latency_p99_ms:.1f} ms")
        print(f"  Latency max   : {self.latency_max_ms:.1f} ms")
        if self.resource_samples:
            cpus = [s["cpu_pct"] for s in self.resource_samples if "cpu_pct" in s]
            rams = [s["ram_mb"]  for s in self.resource_samples if "ram_mb"  in s]
            if cpus:
                print(f"  CPU avg/max   : {sum(cpus)/len(cpus):.1f}% / {max(cpus):.1f}%")
            if rams:
                print(f"  RAM avg/max   : {sum(rams)/len(rams):.1f} MB / {max(rams):.1f} MB")
        print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# Resource monitor (background thread)
# ─────────────────────────────────────────────────────────────

class ResourceMonitor:
    """
    Samples CPU %, RAM usage (MB), and optional GPU utilisation
    at a fixed interval on a daemon thread.
    """

    def __init__(self, interval_s: float = 0.5):
        self.interval  = interval_s
        self.samples: List[Dict] = []
        self._stop     = threading.Event()
        self._thread   = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[Dict]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.samples

    def _sample_loop(self):
        while not self._stop.is_set():
            sample = {"ts": time.time()}
            try:
                import psutil
                sample["cpu_pct"] = psutil.cpu_percent(interval=None)
                sample["ram_mb"]  = psutil.virtual_memory().used / 1e6
            except ImportError:
                pass  # psutil optional

            try:
                import torch
                if torch.cuda.is_available():
                    dev = torch.device("cuda")
                    sample["gpu_alloc_mb"] = torch.cuda.memory_allocated(dev) / 1e6
                    sample["gpu_reserved_mb"] = torch.cuda.memory_reserved(dev) / 1e6
            except ImportError:
                pass

            self.samples.append(sample)
            time.sleep(self.interval)


# ─────────────────────────────────────────────────────────────
# Core load tester
# ─────────────────────────────────────────────────────────────

class LoadTester:
    """
    Exercises an inference callable under controlled load.

    The callable ``inference_fn`` must accept a single image argument
    (PIL Image or file path) and return anything truthy on success.
    All exceptions it raises are captured as failures.

    Usage::

        tester  = LoadTester(inference_fn=pipeline.run)
        report  = tester.run_concurrent(images, n_workers=4, test_name="Stage-1 4T")
        report.print_summary()
        tester.save_report(report, "results/load_test.json")
    """

    def __init__(
        self,
        inference_fn: Callable,
        monitor_resources: bool = True,
        monitor_interval_s: float = 0.5,
    ):
        """
        Args:
            inference_fn: Callable that processes a single image.
            monitor_resources: Whether to sample CPU/RAM/GPU during tests.
            monitor_interval_s: Resource sampling cadence.
        """
        self.fn        = inference_fn
        self.monitor   = monitor_resources
        self._interval = monitor_interval_s

    # ──────────────────────────────────────────────────────────
    # Public test modes
    # ──────────────────────────────────────────────────────────

    def run_concurrent(
        self,
        images: List,
        n_workers: int = 4,
        test_name: str = "concurrent",
    ) -> LoadTestReport:
        """
        Submit ``images`` to ``inference_fn`` using ``n_workers`` threads.

        All images are enqueued immediately; workers race to process them.

        Args:
            images: List of images / file paths to process.
            n_workers: Thread-pool size.
            test_name: Label for the report.

        Returns:
            Populated ``LoadTestReport``.
        """
        monitor = ResourceMonitor(self._interval) if self.monitor else None
        if monitor:
            monitor.start()

        results: List[RequestResult] = []
        t_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(self._run_one, img): img for img in images}
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        duration = time.perf_counter() - t_start
        samples  = monitor.stop() if monitor else []

        return self._build_report(test_name, results, n_workers, duration, samples)

    def measure_throughput(
        self,
        images: List,
        n_workers: int = 1,
        test_name: str = "throughput",
    ) -> LoadTestReport:
        """
        Process ``images`` sequentially (or with ``n_workers`` if > 1) and
        report images-per-second throughput.

        Identical to ``run_concurrent`` but explicitly communicates intent
        of measuring a single-pass throughput benchmark.
        """
        return self.run_concurrent(images, n_workers=n_workers, test_name=test_name)

    def stress_test(
        self,
        image_provider: Callable[[], object],
        duration_s: float = 30.0,
        n_workers: int = 4,
        test_name: str = "stress",
    ) -> LoadTestReport:
        """
        Continuously submit requests from ``image_provider`` for ``duration_s`` seconds.

        Args:
            image_provider: Zero-argument callable that returns one image per call.
                            May be called from multiple threads concurrently.
            duration_s: How long to run (wall-clock seconds).
            n_workers: Thread-pool size.
            test_name: Label for the report.
        """
        stop_event = threading.Event()
        results: List[RequestResult] = []
        lock = threading.Lock()

        monitor = ResourceMonitor(self._interval) if self.monitor else None
        if monitor:
            monitor.start()

        def worker():
            while not stop_event.is_set():
                try:
                    img = image_provider()
                except StopIteration:
                    break
                r = self._run_one(img)
                with lock:
                    results.append(r)

        t_start = time.perf_counter()
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
        for t in threads:
            t.start()

        time.sleep(duration_s)
        stop_event.set()

        for t in threads:
            t.join(timeout=5)

        actual_duration = time.perf_counter() - t_start
        samples = monitor.stop() if monitor else []

        return self._build_report(test_name, results, n_workers, actual_duration, samples)

    def ramp_test(
        self,
        images: List,
        worker_counts: Tuple[int, ...] = (1, 2, 4, 8),
        test_name_prefix: str = "ramp",
    ) -> List[LoadTestReport]:
        """
        Run ``run_concurrent`` for each worker count in ``worker_counts``.

        Useful for identifying the concurrency level at which latency or
        error rate degrades.

        Returns:
            One ``LoadTestReport`` per concurrency level.
        """
        reports = []
        for n in worker_counts:
            print(f"  Ramp test: {n} workers …")
            r = self.run_concurrent(images, n_workers=n, test_name=f"{test_name_prefix}_w{n}")
            reports.append(r)
        return reports

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save_report(
        self,
        report: LoadTestReport,
        path: str = "results/metrics/load_test_report.json",
    ):
        """Serialise the report to JSON."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Load-test report saved → {dest}")

    def save_reports(
        self,
        reports: List[LoadTestReport],
        path: str = "results/metrics/ramp_test_report.json",
    ):
        """Serialise a list of reports (e.g. from ``ramp_test``) to JSON."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w") as f:
            json.dump([r.to_dict() for r in reports], f, indent=2)
        print(f"Ramp-test reports saved → {dest}")

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _run_one(self, image) -> RequestResult:
        t0 = time.perf_counter()
        try:
            self.fn(image)
            latency = (time.perf_counter() - t0) * 1000
            return RequestResult(success=True, latency_ms=latency)
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            return RequestResult(success=False, latency_ms=latency, error_msg=str(exc))

    @staticmethod
    def _build_report(
        name: str,
        results: List[RequestResult],
        n_workers: int,
        duration: float,
        samples: List[Dict],
    ) -> LoadTestReport:
        if not results:
            return LoadTestReport(
                test_name=name, n_requests=0, n_workers=n_workers, duration_s=duration,
                successful=0, failed=0, success_rate=0.0, throughput_rps=0.0,
                latency_mean_ms=0.0, latency_p50_ms=0.0, latency_p95_ms=0.0,
                latency_p99_ms=0.0, latency_max_ms=0.0, latency_min_ms=0.0,
            )

        latencies  = [r.latency_ms for r in results]
        successes  = sum(1 for r in results if r.success)
        n          = len(results)
        sorted_lat = sorted(latencies)

        def pct(p: float) -> float:
            idx = min(int(p * n / 100), n - 1)
            return sorted_lat[idx]

        return LoadTestReport(
            test_name        = name,
            n_requests       = n,
            n_workers        = n_workers,
            duration_s       = round(duration, 3),
            successful       = successes,
            failed           = n - successes,
            success_rate     = successes / n,
            throughput_rps   = round(n / max(duration, 1e-9), 3),
            latency_mean_ms  = round(statistics.mean(latencies), 2),
            latency_p50_ms   = round(pct(50), 2),
            latency_p95_ms   = round(pct(95), 2),
            latency_p99_ms   = round(pct(99), 2),
            latency_max_ms   = round(max(latencies), 2),
            latency_min_ms   = round(min(latencies), 2),
            resource_samples = samples,
        )


# ─────────────────────────────────────────────────────────────
# System health checker (standalone utility)
# ─────────────────────────────────────────────────────────────

class SystemHealthChecker:
    """
    One-shot snapshot of system resources.

    Useful for a quick pre-flight check before running a load test or
    for embedding a health dict into model inference responses.

    Usage::

        checker = SystemHealthChecker()
        health  = checker.check()
        if health["ram_available_gb"] < 2.0:
            print("Warning: low RAM")
    """

    def check(self) -> Dict:
        result = {"timestamp": time.time(), "status": "ok", "warnings": []}

        try:
            import psutil
            vm = psutil.virtual_memory()
            result["cpu_count"]         = psutil.cpu_count()
            result["cpu_pct"]           = psutil.cpu_percent(interval=0.1)
            result["ram_total_gb"]      = round(vm.total / 1e9, 2)
            result["ram_used_gb"]       = round(vm.used  / 1e9, 2)
            result["ram_available_gb"]  = round(vm.available / 1e9, 2)
            result["ram_pct"]           = vm.percent

            if vm.percent > 90:
                result["warnings"].append("High RAM usage (>90%)")
            if result["cpu_pct"] > 90:
                result["warnings"].append("High CPU usage (>90%)")
        except ImportError:
            result["warnings"].append("psutil not installed — resource metrics unavailable")

        try:
            import torch
            result["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                result["cuda_device"]          = torch.cuda.get_device_name(0)
                result["gpu_alloc_mb"]         = round(torch.cuda.memory_allocated() / 1e6, 1)
                result["gpu_reserved_mb"]      = round(torch.cuda.memory_reserved()  / 1e6, 1)
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e6
                result["gpu_total_vram_mb"]    = round(total_vram, 1)
        except ImportError:
            result["warnings"].append("torch not installed — GPU metrics unavailable")

        if result["warnings"]:
            result["status"] = "warning"

        return result

    def print_health(self):
        h = self.check()
        print(f"\n{'='*50}")
        print("System Health Check")
        print(f"{'='*50}")
        for k, v in h.items():
            if k not in ("timestamp", "status", "warnings"):
                print(f"  {k:<25}: {v}")
        if h["warnings"]:
            print("\n  WARNINGS:")
            for w in h["warnings"]:
                print(f"    ⚠  {w}")
        print(f"\n  Status: {h['status'].upper()}")
        print(f"{'='*50}")


# ─────────────────────────────────────────────────────────────
# Convenience runner
# ─────────────────────────────────────────────────────────────

def run_full_load_suite(
    inference_fn: Callable,
    images: List,
    output_dir: str = "results/metrics",
    worker_counts: Tuple[int, ...] = (1, 2, 4),
) -> Dict[str, LoadTestReport]:
    """
    Run a standard benchmark suite and save all reports.

    Steps:
      1. System health check
      2. Single-thread throughput baseline
      3. Ramp test across ``worker_counts``

    Args:
        inference_fn: A callable that processes one image.
        images: Sample images for benchmarking.
        output_dir: Directory for JSON reports.
        worker_counts: Concurrency levels for the ramp test.

    Returns:
        Dict mapping test name → LoadTestReport.
    """
    checker = SystemHealthChecker()
    checker.print_health()

    tester  = LoadTester(inference_fn)
    reports = {}

    # Baseline
    print("\n[1/2] Single-thread throughput baseline …")
    baseline = tester.measure_throughput(images, n_workers=1, test_name="baseline_1w")
    baseline.print_summary()
    tester.save_report(baseline, f"{output_dir}/baseline_report.json")
    reports["baseline"] = baseline

    # Ramp
    print("\n[2/2] Ramp test …")
    ramp_reports = tester.ramp_test(images, worker_counts=worker_counts)
    for r in ramp_reports:
        r.print_summary()
        reports[r.test_name] = r
    tester.save_reports(ramp_reports, f"{output_dir}/ramp_report.json")

    return reports


# ─────────────────────────────────────────────────────────────
# Smoke-test
# ─────────────────────────────────────────────────────────────

def _smoke_test():
    print("Smoke-testing load_tester.py …")

    def fake_inference(img):
        time.sleep(0.01)   # simulate 10 ms inference
        return {"pred": "Urban"}

    images = list(range(20))  # use integers as dummy images
    tester = LoadTester(fake_inference, monitor_resources=False)

    report = tester.run_concurrent(images, n_workers=4, test_name="smoke")
    assert report.n_requests  == 20
    assert report.successful  == 20
    assert report.failed      == 0
    assert report.throughput_rps > 0
    report.print_summary()

    checker = SystemHealthChecker()
    h = checker.check()
    assert "status" in h

    print("  All checks passed.")


if __name__ == "__main__":
    _smoke_test()
