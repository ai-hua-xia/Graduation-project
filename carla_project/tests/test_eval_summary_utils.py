from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.eval_summary_utils import (
    COMPACT_COLUMNS,
    COMPACT_NORM_COLUMNS,
    build_compact_record,
    build_compact_summary_payload,
    build_world_model_config,
    summarize_action_sensitivity,
    resolve_project_path,
)


def _base_result():
    return {
        "name": "wm_demo",
        "ar_psnr": 18.5,
        "ar_ssim": 0.71,
        "ar_lpips": 0.24,
        "ar_fvd": 33.2,
        "action_sens_mean": 12.5,
        "wm_path": "checkpoints/wm/demo/best.pth",
        "token_file": "data/demo/tokens_actions.npz",
        "vqvae_path": "checkpoints/vqvae/demo/best.pth",
        "wm_config": {"context_frames": 4, "tokens_per_frame": 256},
    }


def test_build_compact_record_preserves_existing_mean_and_adds_std():
    source = _base_result()
    action_stats = {
        "action_sensitivity_mean": 99.0,
        "action_sensitivity_std": 3.25,
    }

    record = build_compact_record(
        source,
        action_stats=action_stats,
        preserve_existing_mean=True,
    )

    assert list(record.keys()) == COMPACT_COLUMNS
    assert record["name"] == "wm_demo"
    assert record["action_sens_mean"] == pytest.approx(12.5)
    assert record["action_sensitivity_std"] == pytest.approx(3.25)


def test_build_compact_record_can_replace_mean_with_recomputed_value():
    source = _base_result()
    action_stats = {
        "action_sensitivity_mean": 8.75,
        "action_sensitivity_std": 1.5,
    }

    record = build_compact_record(
        source,
        action_stats=action_stats,
        preserve_existing_mean=False,
    )

    assert record["action_sens_mean"] == pytest.approx(8.75)
    assert record["action_sensitivity_std"] == pytest.approx(1.5)


def test_build_compact_summary_payload_keeps_best_record_and_mode():
    record_a = {
        "name": "wm_b",
        "ar_psnr": 17.0,
        "ar_ssim": 0.62,
        "ar_lpips": 0.30,
        "ar_fvd": 55.0,
        "action_sens_mean": 4.0,
        "action_sensitivity_std": 0.8,
    }
    record_b = {
        "name": "wm_a",
        "ar_psnr": 19.0,
        "ar_ssim": 0.72,
        "ar_lpips": 0.22,
        "ar_fvd": 25.0,
        "action_sens_mean": 6.0,
        "action_sensitivity_std": 1.1,
    }

    payload = build_compact_summary_payload(
        results=[record_a, record_b],
        output_dir="outputs/evaluations/demo",
        source_summary="outputs/evaluations/source/summary.json",
        mode="recompute_mean_std",
    )

    assert payload["output_dir"] == "outputs/evaluations/demo"
    assert payload["source_summary"] == "outputs/evaluations/source/summary.json"
    assert payload["mode"] == "recompute_mean_std"
    assert payload["columns"] == COMPACT_COLUMNS
    assert payload["best"]["name"] == "wm_a"
    assert payload["results"][0]["name"] == "wm_a"
    assert payload["results"][1]["name"] == "wm_b"


def test_build_compact_record_can_emit_normalized_action_columns():
    source = _base_result()
    action_stats = {
        "action_sensitivity_mean": 8.75,
        "action_sensitivity_std": 1.5,
        "action_sensitivity_mean_norm": 8.75 / 256.0,
        "action_sensitivity_std_norm": 1.5 / 256.0,
    }

    record = build_compact_record(
        source,
        action_stats=action_stats,
        preserve_existing_mean=False,
        normalized=True,
    )

    assert list(record.keys()) == COMPACT_NORM_COLUMNS
    assert record["action_sens_mean_norm"] == pytest.approx(8.75 / 256.0)
    assert record["action_sensitivity_std_norm"] == pytest.approx(1.5 / 256.0)


def test_resolve_project_path_expands_relative_paths_under_project_root(tmp_path):
    project_root = tmp_path / "carla_project"
    project_root.mkdir()

    resolved = resolve_project_path(project_root, "data/demo/file.npz")

    assert resolved == project_root / "data/demo/file.npz"
    assert resolve_project_path(project_root, resolved) == resolved


def test_build_world_model_config_prefers_result_config_and_runtime_embeddings():
    default_config = {
        "num_embeddings": 1024,
        "embed_dim": 896,
        "hidden_dim": 1792,
        "num_heads": 8,
        "num_layers": 20,
        "context_frames": 4,
        "action_dim": 2,
        "tokens_per_frame": 1024,
        "use_memory": True,
        "memory_dim": 512,
        "dropout": 0.1,
        "conditioning_type": "adaln_zero",
        "use_action_aux": True,
    }
    result_config = {
        "embed_dim": 512,
        "hidden_dim": 1024,
        "tokens_per_frame": 256,
        "use_memory": False,
        "conditioning_type": "film",
        "use_action_aux": False,
    }

    merged = build_world_model_config(default_config, result_config, num_embeddings=4074)

    assert merged["num_embeddings"] == 4074
    assert merged["embed_dim"] == 512
    assert merged["hidden_dim"] == 1024
    assert merged["tokens_per_frame"] == 256
    assert merged["use_memory"] is False
    assert merged["conditioning_type"] == "film"
    assert merged["use_action_aux"] is False


def test_summarize_action_sensitivity_computes_raw_and_per_token_stats():
    sensitivities = [4.0, 8.0, 12.0]

    summary = summarize_action_sensitivity(sensitivities, tokens_per_frame=4)

    assert summary["action_sensitivity_mean"] == pytest.approx(8.0)
    assert summary["action_sensitivity_std"] == pytest.approx(3.265986323710904)
    assert summary["action_sensitivity_mean_norm"] == pytest.approx(2.0)
    assert summary["action_sensitivity_std_norm"] == pytest.approx(0.816496580927726)
