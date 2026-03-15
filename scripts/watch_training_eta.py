from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

EPOCHS_RE = re.compile(r"^Epochs:\s*(\d+)", re.MULTILINE)
TRAIN_BATCHES_RE = re.compile(r"^Train batches:\s*(\d+)", re.MULTILINE)
VAL_BATCHES_RE = re.compile(r"^Val batches:\s*(\d+)", re.MULTILINE)
PROGRESS_RE = re.compile(
    r"Epoch\s+(\d+)\s+\[(Train|Val)\]:\s+\d+%\|.*?\|\s*(\d+)/(\d+).*?,\s*([0-9]*\.?[0-9]+)it/s"
)


def latest_log() -> Path | None:
    logs = sorted(LOG_DIR.glob("mit_full_training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    epochs_match = EPOCHS_RE.search(text)
    train_match = TRAIN_BATCHES_RE.search(text)
    val_match = VAL_BATCHES_RE.search(text)
    progress_matches = list(PROGRESS_RE.finditer(text))

    if not epochs_match or not train_match or not val_match or not progress_matches:
        return None

    last = progress_matches[-1]
    return {
        "epochs": int(epochs_match.group(1)),
        "train_batches": int(train_match.group(1)),
        "val_batches": int(val_match.group(1)),
        "epoch": int(last.group(1)),
        "phase": last.group(2),
        "done": int(last.group(3)),
        "total": int(last.group(4)),
        "rate": float(last.group(5)),
    }


def format_eta(seconds: int) -> str:
    return str(timedelta(seconds=max(0, seconds))).split(".")[0]


def main() -> int:
    while True:
        log_path = latest_log()
        if log_path is None:
            print("Waiting for training log...", flush=True)
            time.sleep(10)
            continue

        parsed = parse_log(log_path)
        if parsed is None or parsed["rate"] <= 0:
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | waiting for tqdm progress in {log_path.name}", flush=True)
            time.sleep(10)
            continue

        train_total = parsed["train_batches"]
        val_total = parsed["val_batches"]
        current_epoch = parsed["epoch"]
        total_epochs = parsed["epochs"]
        done = parsed["done"]
        rate = parsed["rate"]

        if parsed["phase"] == "Train":
            remaining_this_epoch = (train_total - done) + val_total
        else:
            remaining_this_epoch = val_total - done

        remaining_full_epochs = max(0, total_epochs - current_epoch)
        remaining_all_batches = remaining_this_epoch + remaining_full_epochs * (train_total + val_total)

        epoch_eta_seconds = int(remaining_this_epoch / rate)
        full_eta_seconds = int(remaining_all_batches / rate)

        now = datetime.now()
        epoch_done_at = now + timedelta(seconds=epoch_eta_seconds)
        full_done_at = now + timedelta(seconds=full_eta_seconds)

        print(
            f"{now:%Y-%m-%d %H:%M:%S} | "
            f"epoch={current_epoch}/{total_epochs} phase={parsed['phase']} progress={done}/{parsed['total']} "
            f"rate={rate:.2f} it/s | "
            f"epochETA={format_eta(epoch_eta_seconds)} ({epoch_done_at:%Y-%m-%d %H:%M:%S}) | "
            f"fullETA={format_eta(full_eta_seconds)} ({full_done_at:%Y-%m-%d %H:%M:%S})",
            flush=True,
        )
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())