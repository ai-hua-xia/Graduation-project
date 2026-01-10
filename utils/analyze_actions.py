import argparse
import json
import os
from typing import Dict

import numpy as np


def summarize_actions(actions: np.ndarray) -> Dict[str, float]:
    steer = actions[:, 0]
    throttle = actions[:, 1]
    stats = {
        "count": int(actions.shape[0]),
        "steer_mean": float(np.mean(steer)),
        "steer_std": float(np.std(steer)),
        "steer_abs_mean": float(np.mean(np.abs(steer))),
        "steer_abs_p50": float(np.percentile(np.abs(steer), 50)),
        "steer_abs_p90": float(np.percentile(np.abs(steer), 90)),
        "steer_abs_p95": float(np.percentile(np.abs(steer), 95)),
        "steer_abs_p99": float(np.percentile(np.abs(steer), 99)),
        "throttle_mean": float(np.mean(throttle)),
        "throttle_std": float(np.std(throttle)),
        "throttle_neg_ratio": float(np.mean(throttle < 0)),
        "throttle_zero_ratio": float(np.mean(np.abs(throttle) < 1e-3)),
    }
    return stats


def bucket_ratios(values: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    ratios = {}
    for t in thresholds:
        ratios[f">={t:.2f}"] = float(np.mean(np.abs(values) >= t))
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze action distribution.")
    parser.add_argument(
        "--npz",
        default="dataset_v2_complex/tokens_actions_vqvae_16x16.npz",
        help="Path to tokens/actions npz",
    )
    parser.add_argument(
        "--actions",
        default=None,
        help="Optional actions.npy path (override npz)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output json path",
    )
    args = parser.parse_args()

    if args.actions:
        actions_path = args.actions
        if not os.path.exists(actions_path):
            raise FileNotFoundError(actions_path)
        actions = np.load(actions_path)
    else:
        if not os.path.exists(args.npz):
            raise FileNotFoundError(args.npz)
        data = np.load(args.npz)
        actions = data["actions"]

    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"Expected actions shape (N, 2), got {actions.shape}")

    stats = summarize_actions(actions)
    steer = actions[:, 0]
    throttle = actions[:, 1]

    steer_buckets = bucket_ratios(steer, np.array([0.05, 0.1, 0.2, 0.4, 0.6]))
    brake_buckets = bucket_ratios(-throttle, np.array([0.1, 0.2, 0.4]))

    report = {
        "stats": stats,
        "steer_abs_ratio": steer_buckets,
        "brake_ratio": brake_buckets,
    }

    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
