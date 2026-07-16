"""
DeepSceneLoc — Main Application Launcher
Starts the Phase 2 Modern Web Interface (FastAPI + HTML/JS/CSS)

Usage:
    python app.py
"""
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    print("\n" + "=" * 65)
    print("  🌍 DeepSceneLoc — Phase 2 Modern Web App")
    print("  Visual Scene Understanding + Exact Place Identification")
    print("=" * 65)
    
    # Check for .env file
    if not (ROOT / ".env").exists():
        print("[WARN] .env file not found. Stage 2 (OpenRouter) may not work.")
        print("       Please copy .env.example to .env and add your keys.")

    # Determine Python executable
    # Use venv if available, otherwise use system python
    python_exe = sys.executable
    venv_python = ROOT / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python_exe = str(venv_python)

    print(f"\n[INFO] Starting production server on http://localhost:8000")
    print("[INFO] Press Ctrl+C to stop.\n")

    cmd = [
        python_exe, "-m", "uvicorn",
        "webapp.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info"
    ]

    try:
        subprocess.run(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to start server: {e}")

if __name__ == "__main__":
    main()
