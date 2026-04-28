"""
DeepSceneLoc — App Launcher
Starts the FastAPI production server

Usage:
    venv\Scripts\python.exe app.py
    venv\Scripts\python.exe app.py --port 8080
    venv\Scripts\python.exe app.py --reload     # dev mode
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--port",   type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="Enable hot reload (dev mode)")
    args = p.parse_args()

    python = str(ROOT / "venv" / "Scripts" / "python.exe")

    cmd = [
        python, "-m", "uvicorn",
        "webapp.api:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    print("=" * 60)
    print("  DeepSceneLoc Production App")
    print(f"  URL: http://localhost:{args.port}")
    print(f"  API: http://localhost:{args.port}/api/status")
    print("=" * 60)

    subprocess.run(cmd, cwd=str(ROOT))

if __name__ == "__main__":
    main()
