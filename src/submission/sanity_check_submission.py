"""Generate and validate starter sample submissions for val/test splits."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/submissions"))
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_submission = args.output_dir / "val_sample_submission.csv"
    run_command(
        [
            sys.executable,
            "starter/make_sample_submission_csv.py",
            "--data-root",
            str(args.data_root),
            "--split",
            "val",
            "--output",
            str(val_submission),
        ]
    )
    run_command(
        [
            sys.executable,
            "starter/validate_submission_csv.py",
            "--submission",
            str(val_submission),
            "--data-root",
            str(args.data_root),
            "--split",
            "val",
        ]
    )

    if not args.skip_test:
        test_submission = args.output_dir / "test_sample_submission.csv"
        run_command(
            [
                sys.executable,
                "starter/make_sample_submission_csv.py",
                "--data-root",
                str(args.data_root),
                "--split",
                "test",
                "--output",
                str(test_submission),
            ]
        )
        run_command(
            [
                sys.executable,
                "starter/validate_submission_csv.py",
                "--submission",
                str(test_submission),
                "--data-root",
                str(args.data_root),
                "--split",
                "test",
            ]
        )


if __name__ == "__main__":
    main()
