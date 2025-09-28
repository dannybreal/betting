from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    python = sys.executable
    cmd = [python, "-m", "src.ratings.pipeline", "preview"]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
