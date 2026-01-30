#!/usr/bin/env python3
"""
Check missing episode ranges for Phase A/B.

Usage:
  python tools/check_phase_missing.py --data-dir data/raw_action_corr_v3
  python tools/check_phase_missing.py --data-dir data/raw_action_corr_v3 --phase B
"""

import argparse
import re
from pathlib import Path


def collect_indices(data_dir: Path) -> set[int]:
    idxs: set[int] = set()
    for p in data_dir.glob("episode_*"):
        m = re.search(r"episode_(\d+)", p.name)
        if m:
            idxs.add(int(m.group(1)))
    return idxs


def print_ranges(label: str, missing: list[int]) -> None:
    print(f"{label}: {len(missing)}")
    if not missing:
        return
    print("Missing ranges:")
    start = prev = missing[0]
    for x in missing[1:]:
        if x == prev + 1:
            prev = x
        else:
            print(f"  {start}-{prev}" if start != prev else f"  {start}")
            start = prev = x
    print(f"  {start}-{prev}" if start != prev else f"  {start}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw_action_corr_v3")
    parser.add_argument("--phase", type=str, choices=["A", "B", "ALL"], default="B")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    idxs = collect_indices(data_dir)
    print(f"Total episodes: {len(idxs)}")

    if args.phase in ("A", "ALL"):
        missing_a = [i for i in range(1, 751) if i not in idxs]
        print_ranges("Missing A (1-750)", missing_a)

    if args.phase in ("B", "ALL"):
        missing_b = [i for i in range(751, 1501) if i not in idxs]
        print_ranges("Missing B (751-1500)", missing_b)


if __name__ == "__main__":
    main()
