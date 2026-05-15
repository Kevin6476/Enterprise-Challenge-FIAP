"""Build the demo dashboard and validate MVP artifacts in one command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, cwd=PROJECT_DIR, check=True, stderr=subprocess.STDOUT)


def main() -> None:
    run([sys.executable, "app/build_dashboard.py"])
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run([sys.executable, "scripts/validate_mvp.py"])
    print("\nMVP demo build and validation completed.")


if __name__ == "__main__":
    main()
