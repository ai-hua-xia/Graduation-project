"""
视频生成质量评估指标

包含以下指标：
1. PSNR (Peak Signal-to-Noise Ratio) - 峰值信噪比
2. SSIM (Structural Similarity Index) - 结构相似性
3. LPIPS (Learned Perceptual Image Patch Similarity) - 感知相似性
4. R-FID (Reconstruction FID) - 重建分布距离（可选）
5. FVD (Fréchet Video Distance) - 视频分布距离（可选）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import List, Dict, Tuple, Optional
import cv2


class VideoMetrics:
    """视频质量评估类"""

    def __init__(self, device='cuda', use_lpips=True, use_fid=False, use_fvd=False):
        """
        Args:
            device: 计算设备
            use_lpips: 是否使用LPIPS（需要额外安装lpips库）
            use_fid: 是否计算FID/R-FID（需要torchvision+scipy）
            use_fvd: 是否计算FVD（需要torchvision+scipy）
        """
        self.device = device
        self.use_lpips = use_lpips
        self.lpips_model = None
        self.use_fid = use_fid
        self.fid_model = None
        self.use_fvd = use_fvd
        self.fvd_model = None
        self.fvd_mean = None
        self.fvd_std = None

        if use_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').to(device)
                self.lpips_model.eval()
                print("LPIPS model loaded successfully")
            except ImportError:
                print("Warning: lpips not installed. Run: pip install lpips")
                self.use_lpips = False

        if use_fid:
            try:
                from torchvision.models import inception_v3, Inception_V3_Weights
                weights = Inception_V3_Weights.DEFAULT
                # 新版torchvision要求与weights绑定的aux_logits设置保持一致
                model = inception_v3(weights=weights)
                model.fc = nn.Identity()
                model.to(device)
                model.eval()
                self.fid_model = model
            except Exception as exc:
                print(f"Warning: failed to load Inception for R-FID: {exc}")
                self.use_fid = False

        if use_fvd:
            try:
                from torchvision.models.video import r3d_18, R3D_18_Weights
                weights = R3D_18_Weights.DEFAULT
                model = r3d_18(weights=weights)
                model.fc = nn.Identity()
                model.to(device)
                model.eval()
                self.fvd_model = model
                # Kinetics-400 3D backbone常用归一化参数
                self.fvd_mean = torch.tensor(
                    [0.43216, 0.394666, 0.37645], device=self.device
                ).view(1, 3, 1, 1, 1)
                self.fvd_std = torch.tensor(
                    [0.22803, 0.22145, 0.216989], device=self.device
                ).view(1, 3, 1, 1, 1)
            except Exception as exc:
                print(f"Warning: failed to load video backbone for FVD: {exc}")
                self.use_fvd = False

    def _inception_features(self, images: np.ndarray, batch_size: int = 32) -> np.ndarray:
        if self.fid_model is None:
            return np.empty((0, 2048), dtype=np.float32)

        imgs = images.astype(np.float32) / 255.0
        feats = []
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = torch.from_numpy(imgs[i:i + batch_size]).permute(0, 3, 1, 2).to(self.device)
                if batch.shape[-1] != 299 or batch.shape[-2] != 299:
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                batch = (batch - mean) / std
                out = self.fid_model(batch)
                if isinstance(out, tuple):
                    out = out[0]
                if hasattr(out, "logits"):
                    out = out.logits
                if out.dim() > 2:
                    out = out.flatten(1)
                feats.append(out.cpu().numpy())

        return np.concatenate(feats, axis=0) if feats else np.empty((0, 2048), dtype=np.float32)

    def _video_features(
        self,
        videos: np.ndarray,
        batch_size: int = 8,
        clip_len: int = 16,
    ) -> np.ndarray:
        if self.fvd_model is None:
            return np.empty((0, 512), dtype=np.float32)

        if videos.ndim != 5:
            return np.empty((0, 512), dtype=np.float32)

        vids = videos.astype(np.float32) / 255.0  # (N, T, H, W, C)
        n_videos, timesteps = vids.shape[:2]
        if n_videos == 0 or timesteps < 2:
            return np.empty((0, 512), dtype=np.float32)

        # 统一时间长度，避免不同clip长度导致特征分布不可比
        if timesteps != clip_len:
            indices = np.linspace(0, timesteps - 1, clip_len).astype(np.int64)
            vids = vids[:, indices]

        tensor = torch.from_numpy(vids).permute(0, 4, 1, 2, 3).to(self.device)  # (N, C, T, H, W)
        n, c, t, h, w = tensor.shape
        if h != 112 or w != 112:
            flat = tensor.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w)
            flat = F.interpolate(flat, size=(112, 112), mode='bilinear', align_corners=False)
            tensor = flat.reshape(n, t, c, 112, 112).permute(0, 2, 1, 3, 4)

        tensor = (tensor - self.fvd_mean) / self.fvd_std

        feats = []
        with torch.no_grad():
            for i in range(0, len(tensor), batch_size):
                batch = tensor[i:i + batch_size]
                out = self.fvd_model(batch)
                if out.dim() > 2:
                    out = out.flatten(1)
                feats.append(out.cpu().numpy())

        return np.concatenate(feats, axis=0) if feats else np.empty((0, 512), dtype=np.float32)

    @staticmethod
    def _frechet_distance(feat_1: np.ndarray, feat_2: np.ndarray) -> float:
        if feat_1.size == 0 or feat_2.size == 0:
            return -1.0
        if feat_1.shape[0] < 2 or feat_2.shape[0] < 2:
            return -1.0

        feat_1 = feat_1.astype(np.float64, copy=False)
        feat_2 = feat_2.astype(np.float64, copy=False)
        mu1 = np.mean(feat_1, axis=0)
        mu2 = np.mean(feat_2, axis=0)
        dim = feat_1.shape[1]
        eps = 1e-6

        # 小样本下全协方差矩阵非常不稳定，退化为对角协方差以避免数值爆炸
        if min(feat_1.shape[0], feat_2.shape[0]) <= dim:
            var1 = np.var(feat_1, axis=0) + eps
            var2 = np.var(feat_2, axis=0) + eps
            sigma1 = np.diag(var1)
            sigma2 = np.diag(var2)
        else:
            sigma1 = np.cov(feat_1, rowvar=False)
            sigma2 = np.cov(feat_2, rowvar=False)
            sigma1 = sigma1 + np.eye(dim, dtype=np.float64) * eps
            sigma2 = sigma2 + np.eye(dim, dtype=np.float64) * eps

        diff = mu1 - mu2
        try:
            from scipy.linalg import sqrtm
            covmean = sqrtm(sigma1 @ sigma2)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            if not np.all(np.isfinite(covmean)):
                raise ValueError("non-finite sqrtm output")
        except Exception:
            # 回退到对角近似
            covmean = np.diag(np.sqrt(np.clip(np.diag(sigma1) * np.diag(sigma2), 0.0, None)))

        score = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        if not np.isfinite(score):
            return -1.0
        return float(max(np.real(score), 0.0))

    def compute_rfid(self, pred_frames: np.ndarray, target_frames: np.ndarray) -> float:
        if not self.use_fid or self.fid_model is None:
            return -1.0

        if pred_frames.ndim == 3:
            pred_frames = pred_frames[None, ...]
        if target_frames.ndim == 3:
            target_frames = target_frames[None, ...]

        if len(pred_frames) < 2 or len(target_frames) < 2:
            return -1.0

        feat_pred = self._inception_features(pred_frames)
        feat_target = self._inception_features(target_frames)

        if feat_pred.size == 0 or feat_target.size == 0:
            return -1.0

        return self._frechet_distance(feat_pred, feat_target)

    def compute_fid(self, pred_frames: np.ndarray, target_frames: np.ndarray) -> float:
        """与R-FID一致的帧分布FID，保留独立字段便于与FVD并列展示。"""
        return self.compute_rfid(pred_frames, target_frames)

    def compute_fvd(
        self,
        pred_videos: np.ndarray,
        target_videos: np.ndarray,
        clip_len: int = 16,
        max_videos: Optional[int] = None,
    ) -> float:
        if not self.use_fvd or self.fvd_model is None:
            return -1.0

        if pred_videos.ndim != 5 or target_videos.ndim != 5:
            return -1.0

        n_pred = len(pred_videos)
        n_target = len(target_videos)
        n = min(n_pred, n_target)
        if n < 2:
            return -1.0

        if max_videos is not None:
            n = min(n, max_videos)

        pred = pred_videos[:n]
        target = target_videos[:n]

        feat_pred = self._video_features(pred, clip_len=clip_len)
        feat_target = self._video_features(target, clip_len=clip_len)
        return self._frechet_distance(feat_pred, feat_target)

    def compute_psnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        计算PSNR

        Args:
            pred: 预测图像 (H, W, C) 或 (N, H, W, C)，范围[0, 255]
            target: 真实图像，同上

        Returns:
            PSNR值（dB），越高越好
        """
        # 添加数值稳定性：如果MSE太小，限制PSNR上限
        import warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        if pred.ndim == 4:
            # 多帧，计算平均
            psnrs = []
            for i in range(len(pred)):
                psnr = peak_signal_noise_ratio(target[i], pred[i], data_range=255)
                # 限制PSNR上限为100 dB（MSE < 0.0001时）
                if np.isinf(psnr) or psnr > 100:
                    psnr = 100.0
                psnrs.append(psnr)
            return np.mean(psnrs)
        else:
            psnr = peak_signal_noise_ratio(target, pred, data_range=255)
            if np.isinf(psnr) or psnr > 100:
                psnr = 100.0
            return psnr

    def compute_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        计算SSIM

        Args:
            pred: 预测图像 (H, W, C) 或 (N, H, W, C)，范围[0, 255]
            target: 真实图像，同上

        Returns:
            SSIM值，范围[0, 1]，越高越好
        """
        if pred.ndim == 4:
            ssims = []
            for i in range(len(pred)):
                ssim = structural_similarity(
                    target[i], pred[i],
                    channel_axis=2,
                    data_range=255
                )
                ssims.append(ssim)
            return np.mean(ssims)
        else:
            return structural_similarity(
                target, pred,
                channel_axis=2,
                data_range=255
            )

    def compute_lpips(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        计算LPIPS（感知损失）

        Args:
            pred: 预测图像 (H, W, C) 或 (N, H, W, C)，范围[0, 255]
            target: 真实图像，同上

        Returns:
            LPIPS值，越低越好
        """
        if not self.use_lpips or self.lpips_model is None:
            return -1.0

        def to_tensor(img):
            # (H, W, C) -> (1, C, H, W), 归一化到[-1, 1]
            img = img.astype(np.float32) / 255.0 * 2 - 1
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            return img.to(self.device)

        with torch.no_grad():
            if pred.ndim == 4:
                lpips_values = []
                for i in range(len(pred)):
                    pred_t = to_tensor(pred[i])
                    target_t = to_tensor(target[i])
                    lpips_val = self.lpips_model(pred_t, target_t).item()
                    lpips_values.append(lpips_val)
                return np.mean(lpips_values)
            else:
                pred_t = to_tensor(pred)
                target_t = to_tensor(target)
                return self.lpips_model(pred_t, target_t).item()

    def compute_temporal_consistency(self, frames: np.ndarray) -> float:
        """
        计算时间一致性（相邻帧的变化平滑度）

        Args:
            frames: 视频帧序列 (N, H, W, C)，范围[0, 255]

        Returns:
            时间一致性分数，越低表示越平滑
        """
        if len(frames) < 2:
            return 0.0

        diffs = []
        for i in range(len(frames) - 1):
            diff = np.abs(frames[i+1].astype(float) - frames[i].astype(float))
            diffs.append(np.mean(diff))

        return np.mean(diffs)

    def compute_all_metrics(
        self,
        pred_frames: np.ndarray,
        target_frames: np.ndarray
    ) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            pred_frames: 预测帧序列 (N, H, W, C)，范围[0, 255]
            target_frames: 真实帧序列，同上

        Returns:
            包含所有指标的字典
        """
        metrics = {}

        # 基础指标
        metrics['psnr'] = self.compute_psnr(pred_frames, target_frames)
        metrics['ssim'] = self.compute_ssim(pred_frames, target_frames)

        # LPIPS
        if self.use_lpips:
            metrics['lpips'] = self.compute_lpips(pred_frames, target_frames)

        # 时间一致性
        metrics['temporal_consistency_pred'] = self.compute_temporal_consistency(pred_frames)
        metrics['temporal_consistency_target'] = self.compute_temporal_consistency(target_frames)

        # FID / R-FID（重建分布距离）
        if self.use_fid:
            fid = self.compute_fid(pred_frames, target_frames)
            metrics['fid'] = fid
            metrics['rfid'] = fid  # 兼容旧字段

        return metrics

    def compute_metrics_over_time(
        self,
        pred_frames: np.ndarray,
        target_frames: np.ndarray,
        window_size: int = 10
    ) -> Dict[str, List[float]]:
        """
        计算指标随时间的变化（用于分析误差累积）

        Args:
            pred_frames: 预测帧序列 (N, H, W, C)
            target_frames: 真实帧序列
            window_size: 滑动窗口大小

        Returns:
            每个时间窗口的指标
        """
        n_frames = len(pred_frames)
        n_windows = n_frames // window_size

        metrics_over_time = {
            'psnr': [],
            'ssim': [],
            'frame_idx': []
        }

        if self.use_lpips:
            metrics_over_time['lpips'] = []

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size

            pred_window = pred_frames[start:end]
            target_window = target_frames[start:end]

            metrics_over_time['psnr'].append(
                self.compute_psnr(pred_window, target_window)
            )
            metrics_over_time['ssim'].append(
                self.compute_ssim(pred_window, target_window)
            )
            metrics_over_time['frame_idx'].append(start + window_size // 2)

            if self.use_lpips:
                metrics_over_time['lpips'].append(
                    self.compute_lpips(pred_window, target_window)
                )

        return metrics_over_time


def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """
    从视频文件加载帧

    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数

    Returns:
        帧数组 (N, H, W, C)，BGR格式，范围[0, 255]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return np.array(frames)


def extract_comparison_video_parts(
    comparison_video_path: str,
    max_frames: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从对比视频中提取真实帧和预测帧

    假设对比视频是左右并排的格式（左边真实，右边预测）

    Args:
        comparison_video_path: 对比视频路径
        max_frames: 最大帧数

    Returns:
        (real_frames, pred_frames)，每个都是 (N, H, W, C)
    """
    frames = load_video_frames(comparison_video_path, max_frames)

    if len(frames) == 0:
        raise ValueError(f"Cannot load video: {comparison_video_path}")

    # 假设左右并排
    h, w = frames.shape[1:3]
    mid = w // 2

    real_frames = frames[:, :, :mid, :]
    pred_frames = frames[:, :, mid:, :]

    return real_frames, pred_frames


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate video generation quality')
    parser.add_argument('--comparison-video', type=str, required=True,
                        help='Path to comparison video (side-by-side format)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to evaluate')
    parser.add_argument('--no-lpips', action='store_true',
                        help='Disable LPIPS computation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for metrics (JSON)')

    args = parser.parse_args()

    print(f"Loading video: {args.comparison_video}")
    real_frames, pred_frames = extract_comparison_video_parts(
        args.comparison_video,
        args.max_frames
    )
    print(f"Loaded {len(real_frames)} frames")
    print(f"Frame shape: {real_frames.shape[1:]}")

    # 初始化评估器
    metrics_calculator = VideoMetrics(use_lpips=not args.no_lpips)

    # 计算整体指标
    print("\nComputing metrics...")
    metrics = metrics_calculator.compute_all_metrics(pred_frames, real_frames)

    print("\n" + "="*50)
    print("Overall Metrics:")
    print("="*50)
    print(f"  PSNR:  {metrics['psnr']:.2f} dB (higher is better)")
    print(f"  SSIM:  {metrics['ssim']:.4f} (higher is better, max=1)")
    if 'lpips' in metrics:
        print(f"  LPIPS: {metrics['lpips']:.4f} (lower is better)")
    print(f"  Temporal Consistency (pred):   {metrics['temporal_consistency_pred']:.2f}")
    print(f"  Temporal Consistency (target): {metrics['temporal_consistency_target']:.2f}")

    # 计算随时间变化的指标
    print("\nComputing metrics over time...")
    metrics_over_time = metrics_calculator.compute_metrics_over_time(
        pred_frames, real_frames, window_size=20
    )

    print("\n" + "="*50)
    print("Metrics Over Time (showing degradation):")
    print("="*50)
    for i, frame_idx in enumerate(metrics_over_time['frame_idx']):
        psnr = metrics_over_time['psnr'][i]
        ssim = metrics_over_time['ssim'][i]
        print(f"  Frame {frame_idx:3d}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

    # 保存结果
    if args.output:
        import json
        results = {
            'overall': metrics,
            'over_time': metrics_over_time
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
