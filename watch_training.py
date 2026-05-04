"""
DeepSceneLoc -- Live Training Watcher v4
- Reads UTF-16 training.log (PowerShell Tee-Object format)
- Reads file HEADER for start timestamp, TAIL for live tqdm
- Shows live epoch, phase (Train/Val), batch progress, best acc, ETA
Run: python watch_training.py
"""
import re, time, torch
from pathlib import Path
from datetime import datetime, timedelta

PROJECT   = Path(__file__).parent
LOG       = PROJECT / "training.log"
REFRESH   = 15      # seconds between screen refresh
BATCHES   = 4525    # train batches per epoch
TOTAL_EP  = 40

def get_checkpoint_path(model_name):
    if "vit" in model_name.lower():
        return PROJECT / "models" / "checkpoints" / "vit" / "ViT-B_16_best.pth"
    return PROJECT / "models" / "checkpoints" / "efficientnet" / "EfficientNet-B0_best.pth"
REFRESH   = 15      # seconds between screen refresh
BATCHES   = 4525    # train batches per epoch
TOTAL_EP  = 40

# ── I/O helpers ───────────────────────────────────────────────────────────────

def decode_log(raw: bytes) -> str:
    """Strip null bytes from UTF-16LE (PowerShell Tee-Object) and decode."""
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode('utf-16', errors='ignore')
    return raw.replace(b'\x00', b'').decode('utf-8', errors='ignore')

def read_log_head(n_bytes=12000):
    """Read the first N bytes (contains training config header)."""
    try:
        with LOG.open('rb') as f:
            return decode_log(f.read(n_bytes)).replace('\r\n', '\n').replace('\r', '\n').splitlines()
    except Exception:
        return []

def read_log_tail(n_lines=600):
    """Read the last N lines (contains live tqdm and BEST lines)."""
    try:
        raw   = LOG.read_bytes()
        text  = decode_log(raw).replace('\r\n', '\n').replace('\r', '\n')
        lines = text.splitlines()
        return lines[-n_lines:]
    except Exception:
        return []

# ── parsers ───────────────────────────────────────────────────────────────────

def get_training_info(head_lines):
    """Extract training start timestamp and model name from header."""
    start_ts = None
    model_name = "EfficientNet-B0"
    for l in head_lines:
        m_model = re.search(r'Model\s+:\s+(.+)', l)
        if m_model:
            model_name = m_model.group(1).strip()
            
        m_ts = re.search(r'(EfficientNet-B0|ViT-B_16)_(\d{8})_(\d{6})', l)
        if m_ts:
            try:
                start_ts = datetime.strptime(m_ts.group(2) + m_ts.group(3), '%Y%m%d%H%M%S')
            except Exception:
                pass
    return start_ts, model_name

def get_tqdm_progress(tail_lines):
    """Return latest tqdm progress dict: phase, pct, done, total, el_s, rem_s, speed."""
    pattern = re.compile(
        r'(Train|Val):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s+\[(\d+):(\d+)<(\d+):(\d+),\s*([\d.]+)it/s'
    )
    for l in reversed(tail_lines):
        m = pattern.search(l)
        if m:
            return {
                'phase' : m.group(1),
                'pct'   : int(m.group(2)),
                'done'  : int(m.group(3)),
                'total' : int(m.group(4)),
                'el_s'  : int(m.group(5)) * 60 + int(m.group(6)),
                'rem_s' : int(m.group(7)) * 60 + int(m.group(8)),
                'speed' : float(m.group(9)),
            }
    return None

def get_bests(tail_lines):
    """Return list of val_acc floats from [BEST] lines in log tail."""
    bests = []
    for l in tail_lines:
        m = re.search(r'\[BEST\].*val_acc=([\d.]+)', l)
        if m:
            bests.append(float(m.group(1)))
    return bests

def get_best_checkpoint(ckpt_path):
    try:
        d = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        return d.get('epoch', 0), round(d.get('val_acc', 0) * 100, 4)
    except Exception:
        return 0, 0.0

# ── main ──────────────────────────────────────────────────────────────────────

# One-time: read header for start timestamp
head_lines = read_log_head(12000)
start_ts, model_name   = get_training_info(head_lines)

ckpt_path = get_checkpoint_path(model_name)

print("=" * 72)
print(f"  DeepSceneLoc   {model_name}   Live Watcher v4")
print(f"  training.log  |  Refresh {REFRESH}s  |  Ctrl+C to stop")
print("=" * 72)
print()

if start_ts:
    print(f"  Training started : {start_ts.strftime('%Y-%m-%d %H:%M:%S')} (local)")
    print()
else:
    print("  (Could not find start timestamp in log header)")
    print()

known_bests = []

try:
    while True:
        tail      = read_log_tail(600)
        tqdm_info = get_tqdm_progress(tail)
        bests     = get_bests(tail)
        ck_ep, ck_acc = get_best_checkpoint(ckpt_path)
        ts        = datetime.now().strftime('%H:%M:%S')

        # ── Epoch number ─────────────────────────────────────────────────────
        if start_ts:
            elapsed_s   = (datetime.now() - start_ts).total_seconds()
            EP_SEC      = 500.0   # ~8.3 min/epoch measured
            ep_estimate = min(int(elapsed_s / EP_SEC) + 1, TOTAL_EP)
            elapsed_str = str(timedelta(seconds=int(elapsed_s)))
        else:
            ep_estimate = ck_ep + 1
            elapsed_str = '?'

        # ── Announce new bests ───────────────────────────────────────────────
        for acc in bests[len(known_bests):]:
            delta = f"+{(acc - known_bests[-1]) * 100:.4f}%" if known_bests else "start"
            print(f"\n  >> NEW BEST  val_acc={acc*100:.4f}%  ({delta})"
                  f"  Epoch ~{ep_estimate}")
        known_bests = bests[:]

        # ── Progress bar ─────────────────────────────────────────────────────
        if tqdm_info:
            p       = tqdm_info
            ep_tot  = p['el_s'] + p['rem_s']
            bar     = '#' * (p['pct'] // 5) + '-' * (20 - p['pct'] // 5)
            phase   = p['phase']
            ep_left = max(TOTAL_EP - ep_estimate, 0)
            eta_s   = p['rem_s'] + ep_left * ep_tot
            eta_str = (datetime.now() + timedelta(seconds=eta_s)).strftime('%H:%M')
            line = (
                f"[{ts}]  Ep {ep_estimate:2d}/{TOTAL_EP}  "
                f"[{phase}] |{bar}| {p['pct']:3d}%  "
                f"{p['done']:4d}/{p['total']}  "
                f"{p['speed']:.1f}it/s  "
                f"~{ep_tot//60}m{ep_tot%60:02d}s/ep  "
                f"Best={ck_acc:.4f}%  "
                f"Elapsed={elapsed_str}  "
                f"ETA={eta_str}"
            )
        else:
            line = (
                f"[{ts}]  Ep ~{ep_estimate}/{TOTAL_EP}  "
                f"Best={ck_acc:.4f}%  Elapsed={elapsed_str}  (reading log...)"
            )

        print(f"\r{line:<120}", end='', flush=True)
        time.sleep(REFRESH)

except KeyboardInterrupt:
    ck_ep, ck_acc = get_best_checkpoint(ckpt_path)
    print(f"\n\n{'='*60}")
    print(f"  Training running in background -- do not close its terminal")
    print(f"  Best so far  : {ck_acc:.4f}%  at checkpoint epoch={ck_ep}")
    if start_ts:
        elapsed = datetime.now() - start_ts
        ep_done = int(elapsed.total_seconds() / 500)
        print(f"  Elapsed      : {str(elapsed).split('.')[0]}  (~{ep_done} epochs done)")
    print(f"  Restart watch: python watch_training.py")
    print(f"{'='*60}")
