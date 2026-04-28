"""
Watch Training Progress -- DeepSceneLoc
Reads CHECKPOINT after each epoch (not history which is written only at end).
Run from any directory: python watch_training.py
"""
import json, time, torch
from pathlib import Path
from datetime import datetime

PROJECT = Path(r"D:\Final Project\DeepSceneLoc-Visual-Scene-Understanding-for-Image-Based-Location-Estimation-Using-Deep-Learning")
CKPT    = PROJECT / "models" / "checkpoints" / "efficientnet" / "EfficientNet-B0_best.pth"
HIST    = PROJECT / "logs" / "efficientnet" / "EfficientNet-B0_history.json"
REFRESH = 60  # seconds

print("=" * 55)
print("  DeepSceneLoc -- EfficientNet-B0 Live Watcher")
print(f"  Refresh : every {REFRESH}s  |  Ctrl+C to stop")
print("=" * 55)

last_epoch = 0
epoch_start = time.time()

try:
    while True:
        ts = datetime.now().strftime("%H:%M:%S")

        # Primary source: checkpoint (written every time best improves)
        ckpt_epoch, ckpt_acc = 0, 0.0
        if CKPT.exists():
            try:
                d = torch.load(str(CKPT), map_location="cpu", weights_only=False)
                ckpt_epoch = d.get("epoch", 0)
                ckpt_acc   = d.get("val_acc", 0.0)
            except Exception:
                pass

        # Secondary source: history (only available after training ends)
        hist_epochs = 0
        if HIST.exists():
            try:
                h = json.loads(HIST.read_text())
                # Only trust if mtime is recent (modified AFTER training started)
                hist_epochs = len(h.get("val_acc", []))
            except Exception:
                pass

        elapsed = int(time.time() - epoch_start)
        mins, secs = divmod(elapsed, 60)

        if ckpt_epoch > last_epoch:
            # New best epoch completed
            print(f"\n[{ts}] Epoch {ckpt_epoch}/40 DONE -- Best val: {ckpt_acc*100:.2f}%  [NEW BEST]")
            last_epoch = ckpt_epoch
            epoch_start = time.time()
        else:
            # Epoch still in progress
            print(
                f"[{ts}] Epoch {ckpt_epoch+1}/40 in progress... "
                f"({mins}m{secs:02d}s since last best)  |  "
                f"Best so far: {ckpt_acc*100:.2f}% at epoch {ckpt_epoch}",
                end="\r", flush=True
            )

        time.sleep(REFRESH)

except KeyboardInterrupt:
    print(f"\n\nWatcher stopped. Best result: {ckpt_acc*100:.2f}% at epoch {ckpt_epoch}/40")
