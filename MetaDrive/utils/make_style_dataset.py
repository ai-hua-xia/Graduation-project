import argparse
import glob
import os
import shutil

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create night/fog/snow style dataset from RGB images.")
    parser.add_argument(
        "--input",
        default="dataset_v2_complex/images/*.png",
        help="Input images glob.",
    )
    parser.add_argument(
        "--output-root",
        default="dataset_style",
        help="Output root directory. Each style will be under <root>/<style>/images.",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        default=["night", "fog", "snow"],
        help="Styles to generate: night fog snow",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=-1)
    parser.add_argument(
        "--copy-actions",
        action="store_true",
        help="Copy actions.npy into each style folder if found.",
    )
    return parser.parse_args()


def apply_night(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.6, 0.9)
    hsv[..., 2] *= rng.uniform(0.25, 0.5)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-3, 3)) % 180
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    out[..., 2] = np.clip(out[..., 2] * rng.uniform(1.05, 1.2), 0, 1)
    out[..., 0] = np.clip(out[..., 0] * rng.uniform(0.8, 0.95), 0, 1)
    out[..., 1] = np.clip(out[..., 1] * rng.uniform(0.85, 1.0), 0, 1)
    return (out * 255).astype(np.uint8)


def apply_fog(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    h, w = img_f.shape[:2]
    fog_color = np.array([0.85, 0.88, 0.92], dtype=np.float32)
    fog_strength = rng.uniform(0.3, 0.6)
    power = rng.uniform(1.5, 2.5)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    alpha = fog_strength * (y ** power)
    alpha = np.repeat(alpha, w, axis=1)[:, :, None]
    out = img_f * (1 - alpha) + fog_color * alpha
    contrast = rng.uniform(0.6, 0.85)
    out = (out - 0.5) * contrast + 0.5
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def apply_snow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    h, w = img_f.shape[:2]
    snow = np.zeros_like(img_f, dtype=np.float32)
    num_flakes = int(h * w * rng.uniform(0.0006, 0.0012))
    for _ in range(num_flakes):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        r = int(rng.integers(1, 3))
        cv2.circle(snow, (x, y), r, (1.0, 1.0, 1.0), -1)
    if rng.random() < 0.6:
        for _ in range(num_flakes // 4):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            length = int(rng.integers(5, 15))
            cv2.line(snow, (x, y), (x + length, y + length), (1.0, 1.0, 1.0), 1)
    alpha = rng.uniform(0.2, 0.35)
    out = img_f * (1 - alpha) + snow * alpha
    out[..., 2] = np.clip(out[..., 2] * rng.uniform(1.02, 1.08), 0, 1)
    out[..., 0] = np.clip(out[..., 0] * rng.uniform(0.95, 1.02), 0, 1)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


STYLE_FUNCS = {
    "night": apply_night,
    "fog": apply_fog,
    "snow": apply_snow,
}


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.input))
    if not files:
        print("No images found. Check --input.")
        return

    if args.max_images > 0:
        files = files[: args.max_images]

    missing = [s for s in args.styles if s not in STYLE_FUNCS]
    if missing:
        raise ValueError(f"Unknown styles: {missing}")

    actions_path = os.path.join(os.path.dirname(os.path.dirname(args.input)), "actions.npy")
    if args.copy_actions and not os.path.exists(actions_path):
        print(f"actions.npy not found at {actions_path}, skip copying.")
        actions_path = None

    for style in args.styles:
        out_dir = os.path.join(args.output_root, style, "images")
        os.makedirs(out_dir, exist_ok=True)
        if args.copy_actions and actions_path:
            shutil.copy2(actions_path, os.path.join(args.output_root, style, "actions.npy"))

    for i, path in enumerate(files):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng(args.seed + i)
        for style in args.styles:
            out = STYLE_FUNCS[style](img, rng)
            out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(args.output_root, style, "images", os.path.basename(path))
            cv2.imwrite(out_path, out_bgr)
        if i % 500 == 0:
            print(f"Processed {i}/{len(files)}")

    print(f"Done. Output root: {args.output_root}")


if __name__ == "__main__":
    main()
