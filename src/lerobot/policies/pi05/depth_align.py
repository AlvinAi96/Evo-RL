#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor, nn

from lerobot.policies.pi05.configuration_pi05 import PI05DepthAlignConfig

PI05_DEPTH_TARGETS_KEY = "observation.depth_align.targets"


class PI05DepthTargetGenerator:
    """用冻结的 MoGe + LingBot-Depth 为 RGB 图像生成 depth feature target。"""

    def __init__(self, config: PI05DepthAlignConfig, device: torch.device):
        if not config.moge_path or not config.lingbot_depth_path:
            raise ValueError(
                "depth_align.enable=true requires both depth_align.moge_path and "
                "depth_align.lingbot_depth_path."
            )

        try:
            from mdm.model.v2 import MDMModel
            from moge.model.v2 import MoGeModel
        except ImportError as exc:
            raise ImportError(
                "Depth alignment requires MoGe and LingBot-Depth. Install their packages first, "
                "for example with editable installs of the LingBot-VLA submodules."
            ) from exc

        self.config = config
        self.device = device
        self.moge_model = MoGeModel.from_pretrained(config.moge_path)
        self.lingbot_depth_model = MDMModel.from_pretrained(config.lingbot_depth_path)

        # teacher 只提供监督信号，不能被 optimizer 更新。
        for model in (self.moge_model, self.lingbot_depth_model):
            for param in model.parameters():
                param.requires_grad = False
            model.to(device)
            model.eval()

    @torch.no_grad()
    def __call__(self, images: list[Tensor]) -> Tensor:
        """把 Pi0.5 已 resize/normalize 的多相机图像转成 LingBot-Depth feature target。"""
        if not images:
            raise ValueError("Depth alignment needs at least one image tensor.")

        image_stack = torch.stack(images, dim=1)
        bsize, num_cameras = image_stack.shape[:2]
        flat_images = image_stack.reshape(bsize * num_cameras, *image_stack.shape[2:])

        # Pi0.5 图像进入 policy 前是 [-1, 1]；teacher 需要 [0, 1] RGB。
        input_images = ((flat_images.to(self.device, dtype=torch.float32) + 1.0) / 2.0).clamp(0.0, 1.0)
        output_moge = self.moge_model.infer(
            input_images,
            resolution_level=self.config.resolution_level,
            num_tokens=self.config.target_num_tokens,
            apply_mask=False,
        )
        depth_pred = output_moge["depth"].detach().clone()
        if depth_pred.ndim == 4 and depth_pred.shape[1] == 1:
            depth_pred = depth_pred[:, 0]
        depth_pred = torch.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0)

        depth_target, _ = self.lingbot_depth_model.infer_feat(
            input_images,
            depth_pred,
            depth_down_scale=1,
            resolution_level=self.config.resolution_level,
            num_tokens=self.config.target_num_tokens,
            enable_depth_mask=False,
        )
        if depth_target.ndim != 4:
            raise ValueError(f"Unexpected LingBot-Depth target shape: {tuple(depth_target.shape)}")

        # LingBot-Depth 输出 [B*N, D, H, W]，训练 loss 使用 [B*N, H*W, D]。
        depth_target = depth_target.permute(0, 2, 3, 1).contiguous()
        return depth_target.view(depth_target.shape[0], -1, depth_target.shape[-1])


def make_depth_align_head(hidden_dim: int, target_dim: int) -> nn.Module:
    """构造轻量投影头，把 VLM 图像 token 映射到 LingBot-Depth feature 维度。"""
    return nn.Sequential(
        nn.Linear(hidden_dim, target_dim * 2),
        nn.GELU(),
        nn.Linear(target_dim * 2, target_dim),
    )


def resize_token_grid(tokens: Tensor, target_token_size: int) -> Tensor:
    """将任意方形 token 网格缩放到 depth target 的 token 网格大小。"""
    token_count = tokens.shape[1]
    source_token_size = int(token_count**0.5)
    target_count = target_token_size * target_token_size
    if token_count == target_count:
        return tokens
    if source_token_size * source_token_size != token_count:
        resized = F.interpolate(
            tokens.transpose(1, 2),
            size=target_count,
            mode="linear",
            align_corners=False,
        )
        return resized.transpose(1, 2)

    grid = tokens.view(tokens.shape[0], source_token_size, source_token_size, tokens.shape[-1])
    grid = grid.permute(0, 3, 1, 2).contiguous()
    grid = F.interpolate(
        grid,
        size=(target_token_size, target_token_size),
        mode="bilinear",
        align_corners=False,
    )
    return (
        grid.permute(0, 2, 3, 1)
        .contiguous()
        .view(tokens.shape[0], target_count, tokens.shape[-1])
    )


def compute_depth_alignment_loss(
    prefix_hidden_states: Tensor,
    image_token_counts: list[int],
    img_masks: list[Tensor],
    depth_targets: Tensor,
    depth_align_head: nn.Module,
    config: PI05DepthAlignConfig,
) -> tuple[Tensor, Tensor]:
    """计算 policy 图像 hidden states 与 LingBot-Depth teacher feature 的辅助对齐损失。"""
    if depth_targets.ndim == 4:
        depth_targets = depth_targets.reshape(-1, depth_targets.shape[-2], depth_targets.shape[-1])
    if depth_targets.ndim != 3:
        raise ValueError(f"depth_targets must be [B*N,L,D], got {tuple(depth_targets.shape)}.")

    preds: list[Tensor] = []
    start = 0
    head_param = next(depth_align_head.parameters())
    for token_count in image_token_counts:
        image_states = prefix_hidden_states[:, start : start + token_count]
        image_states = resize_token_grid(image_states, config.target_token_size)
        # depth head 默认 fp32；这里显式对齐 dtype/device，兼容 Pi0.5 bf16 训练。
        image_states = image_states.to(device=head_param.device, dtype=head_param.dtype)
        preds.append(depth_align_head(image_states).to(dtype=torch.float32))
        start += token_count

    depth_preds = torch.stack(preds, dim=1).reshape(-1, preds[0].shape[1], preds[0].shape[2])
    flat_masks = torch.stack(img_masks, dim=1).reshape(-1).to(device=depth_preds.device, dtype=torch.bool)

    if depth_targets.shape[0] != depth_preds.shape[0]:
        raise ValueError(
            "Depth target camera count mismatch: "
            f"pred={depth_preds.shape[0]}, target={depth_targets.shape[0]}."
        )
    if depth_targets.shape[1:] != depth_preds.shape[1:]:
        raise ValueError(
            "Depth target shape mismatch: "
            f"pred={tuple(depth_preds.shape)}, target={tuple(depth_targets.shape)}."
        )

    if not bool(flat_masks.any()):
        return depth_preds.sum() * 0.0, depth_preds

    depth_targets = depth_targets.to(device=depth_preds.device, dtype=depth_preds.dtype)
    loss = F.smooth_l1_loss(
        depth_preds[flat_masks],
        depth_targets[flat_masks].detach(),
        reduction="mean",
    )
    return loss, depth_preds


def _tensor_to_rgb_image(image_tensor: Tensor) -> Image.Image:
    """把 [-1,1] 或 [0,1] 的 CHW tensor 转成 RGB 图片，供可视化拼图使用。"""
    img = image_tensor.detach().float().cpu()
    if img.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got {tuple(img.shape)}.")
    if img.min() < 0:
        img = (img + 1.0) / 2.0
    img = img.clamp(0.0, 1.0)
    img_np = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(img_np, mode="RGB")


def _feature_norm_image(features: Tensor, token_size: int) -> Image.Image:
    """把 depth feature 的 L2 norm 渲染成灰度热力图，方便快速肉眼检查。"""
    feat = features.detach().float().cpu()
    heat = torch.linalg.vector_norm(feat, dim=-1)
    heat = heat.view(token_size, token_size)
    heat = heat - heat.min()
    heat = heat / heat.max().clamp_min(1e-6)
    heat_np = (heat.numpy() * 255.0).astype(np.uint8)
    return (
        Image.fromarray(heat_np, mode="L")
        .resize((224, 224), Image.Resampling.BILINEAR)
        .convert("RGB")
    )


def save_depth_alignment_visualization(
    images: list[Tensor],
    depth_preds: Tensor,
    depth_targets: Tensor,
    img_masks: list[Tensor],
    output_dir: str,
    step: int,
    max_items: int,
    token_size: int,
) -> list[Path]:
    """保存 RGB / teacher target / policy pred 拼图；只用于调试，不参与训练图。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_stack = torch.stack(images, dim=1)
    flat_images = image_stack.reshape(-1, *image_stack.shape[2:])
    flat_masks = torch.stack(img_masks, dim=1).reshape(-1).detach().cpu().bool()
    depth_targets = depth_targets.reshape(depth_preds.shape)

    written: list[Path] = []
    valid_indices = torch.nonzero(flat_masks, as_tuple=False).flatten().tolist()
    for idx in valid_indices[:max_items]:
        rgb = _tensor_to_rgb_image(flat_images[idx])
        target = _feature_norm_image(depth_targets[idx], token_size)
        pred = _feature_norm_image(depth_preds[idx], token_size)

        canvas = Image.new(
            "RGB",
            (rgb.width + target.width + pred.width, rgb.height),
            color=(255, 255, 255),
        )
        canvas.paste(rgb, (0, 0))
        canvas.paste(target, (rgb.width, 0))
        canvas.paste(pred, (rgb.width + target.width, 0))

        path = out_dir / f"depth_align_step_{step:08d}_{idx:02d}.png"
        canvas.save(path)
        written.append(path)
    return written
