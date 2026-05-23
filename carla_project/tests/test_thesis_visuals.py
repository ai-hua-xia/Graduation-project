from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.thesis_figures.make_thesis_visuals import build_autoregressive_frame_samples
from tools.thesis_figures.make_thesis_visuals import rank_vqvae_reconstruction_candidates
from tools.thesis_figures.make_thesis_visuals import select_detail_crop_box
from tools.thesis_figures.make_thesis_visuals import vqvae_reconstruction_layout


def test_autoregressive_samples_map_hundred_frame_label_to_last_video_index():
    samples = build_autoregressive_frame_samples(frame_count=100)

    assert samples == [
        (0, "第零帧"),
        (20, "第二十帧"),
        (40, "第四十帧"),
        (60, "第六十帧"),
        (80, "第八十帧"),
        (99, "第一百帧"),
    ]


def test_detail_crop_box_prefers_textured_difference_region():
    orig = np.zeros((128, 128, 3), dtype=np.uint8)
    recon = orig.copy()

    # Smooth bright sky-like area should not be selected over a detailed patch.
    orig[:64, :64] = 220
    recon[:64, :64] = 218

    textured = ((np.indices((48, 48)).sum(axis=0) % 2) * 255).astype(np.uint8)
    orig[70:118, 70:118] = np.repeat(textured[:, :, None], 3, axis=2)
    recon[70:118, 70:118] = 128

    x0, y0, x1, y1 = select_detail_crop_box(orig, recon, crop_size=48, stride=16)

    assert x0 >= 64
    assert y0 >= 64
    assert x1 <= 128
    assert y1 <= 128


def test_rank_reconstruction_candidates_keeps_structure_and_detail_loss():
    candidates = [
        {
            "path": "low_detail.png",
            "psnr": 34.0,
            "ssim": 0.94,
            "detail_loss": 1.0,
            "crop_texture": 2.0,
        },
        {
            "path": "broken_structure.png",
            "psnr": 20.0,
            "ssim": 0.55,
            "detail_loss": 50.0,
            "crop_texture": 60.0,
        },
        {
            "path": "good_demo.png",
            "psnr": 29.0,
            "ssim": 0.88,
            "detail_loss": 18.0,
            "crop_texture": 28.0,
        },
    ]

    ranked = rank_vqvae_reconstruction_candidates(candidates, min_ssim=0.8, min_psnr=24.0)

    assert [item["path"] for item in ranked] == ["good_demo.png", "low_detail.png"]


def test_vqvae_reconstruction_layout_uses_requested_spacing_ratios():
    layout = vqvae_reconstruction_layout()
    col_x = layout["col_x"]
    img_w = layout["img_w"]
    img_h = layout["img_h"]
    full_pair_gap = col_x[1] - (col_x[0] + img_w)
    full_to_detail_gap = col_x[2] - (col_x[1] + img_w)
    title_to_header_gap = layout["title_y"] - layout["header_y"]
    header_to_image_gap = layout["header_y"] - (layout["row_y"][0] + img_h)

    assert title_to_header_gap == pytest.approx(0.06)
    assert full_to_detail_gap == pytest.approx(full_pair_gap * 1.5)
    assert header_to_image_gap == pytest.approx(title_to_header_gap * 1.5)
