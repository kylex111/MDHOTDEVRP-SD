"""Project entry point; see experiments.cli for arguments and commands."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from experiments.cli import main

    main()
