import copy
from pathlib import Path

import numpy as np


COMPACT_COLUMNS = [
    "name",
    "ar_psnr",
    "ar_ssim",
    "ar_lpips",
    "ar_fvd",
    "action_sens_mean",
    "action_sensitivity_std",
]

COMPACT_NORM_COLUMNS = [
    "name",
    "ar_psnr",
    "ar_ssim",
    "ar_lpips",
    "ar_fvd",
    "action_sens_mean_norm",
    "action_sensitivity_std_norm",
]


def sort_compact_results(results, action_mean_key="action_sens_mean"):
    return sorted(
        results,
        key=lambda row: (
            -float(row["ar_psnr"]),
            -float(row["ar_ssim"]),
            float(row["ar_lpips"]),
            float(row["ar_fvd"]),
            -float(row[action_mean_key]),
        ),
    )


def build_compact_record(source_result, action_stats, preserve_existing_mean, normalized=False):
    if action_stats is None or "action_sensitivity_std" not in action_stats:
        raise ValueError("action_stats with action_sensitivity_std is required")

    if normalized:
        if preserve_existing_mean:
            mean_value = float(source_result["action_sens_mean"]) / float(
                extract_tokens_per_frame(source_result)
            )
        else:
            mean_value = action_stats["action_sensitivity_mean_norm"]
        std_value = action_stats["action_sensitivity_std_norm"]
        record = {
            "name": source_result["name"],
            "ar_psnr": float(source_result["ar_psnr"]),
            "ar_ssim": float(source_result["ar_ssim"]),
            "ar_lpips": float(source_result["ar_lpips"]),
            "ar_fvd": float(source_result["ar_fvd"]),
            "action_sens_mean_norm": float(mean_value),
            "action_sensitivity_std_norm": float(std_value),
        }
    else:
        if preserve_existing_mean:
            mean_value = source_result["action_sens_mean"]
        else:
            mean_value = action_stats["action_sensitivity_mean"]
        record = {
            "name": source_result["name"],
            "ar_psnr": float(source_result["ar_psnr"]),
            "ar_ssim": float(source_result["ar_ssim"]),
            "ar_lpips": float(source_result["ar_lpips"]),
            "ar_fvd": float(source_result["ar_fvd"]),
            "action_sens_mean": float(mean_value),
            "action_sensitivity_std": float(action_stats["action_sensitivity_std"]),
        }
    return record


def build_compact_summary_payload(
    results,
    output_dir,
    source_summary,
    mode,
    columns=COMPACT_COLUMNS,
    action_mean_key="action_sens_mean",
):
    sorted_results = sort_compact_results(results, action_mean_key=action_mean_key)
    best = copy.deepcopy(sorted_results[0]) if sorted_results else None
    return {
        "output_dir": output_dir,
        "source_summary": source_summary,
        "mode": mode,
        "columns": columns,
        "num_results": len(sorted_results),
        "best": best,
        "results": copy.deepcopy(sorted_results),
    }


def resolve_project_path(project_root, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def build_world_model_config(default_config, result_config, num_embeddings):
    config = dict(default_config)
    if result_config:
        config.update(result_config)
    config["num_embeddings"] = int(num_embeddings)
    return config


def extract_tokens_per_frame(source_result):
    wm_config = source_result.get("wm_config") or {}
    tokens_per_frame = wm_config.get("tokens_per_frame")
    if tokens_per_frame is None:
        raise ValueError(f"missing wm_config.tokens_per_frame for {source_result.get('name')}")
    return int(tokens_per_frame)


def summarize_action_sensitivity(sensitivities, tokens_per_frame):
    values = np.asarray(sensitivities, dtype=np.float64)
    if values.size == 0:
        raise ValueError("sensitivities must not be empty")
    if tokens_per_frame <= 0:
        raise ValueError("tokens_per_frame must be positive")

    raw_mean = float(np.mean(values))
    raw_std = float(np.std(values))
    norm_values = values / float(tokens_per_frame)

    return {
        "action_sensitivity_mean": raw_mean,
        "action_sensitivity_std": raw_std,
        "action_sensitivity_mean_norm": float(np.mean(norm_values)),
        "action_sensitivity_std_norm": float(np.std(norm_values)),
    }
