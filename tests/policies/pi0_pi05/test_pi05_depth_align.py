#!/usr/bin/env python

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.pi0.modeling_pi0 import resize_with_pad_torch as pi0_resize_with_pad
from lerobot.policies.pi0_fast.modeling_pi0_fast import resize_with_pad_torch as pi0_fast_resize_with_pad
from lerobot.policies.pi05.configuration_pi05 import PI05Config, PI05DepthAlignConfig
from lerobot.policies.pi05.depth_align import (
    PI05DepthTargetGenerator,
    compute_depth_alignment_loss,
    save_depth_alignment_visualization,
)
from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch


def test_pi05_depth_align_defaults_to_disabled():
    cfg = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    assert cfg.depth_align.enable is False
    assert cfg.depth_align.visualize.enable is False


@pytest.mark.parametrize(
    "resize",
    [resize_with_pad_torch, pi0_resize_with_pad, pi0_fast_resize_with_pad],
)
def test_pi_family_float_resize_uses_zero_before_normalization(resize):
    image = torch.ones(1, 3, 2, 4, dtype=torch.float32)

    resized = resize(image, 4, 4)
    normalized = resized * 2.0 - 1.0

    assert resized.min().item() == 0.0
    assert normalized.min().item() == -1.0
    assert normalized.max().item() == 1.0


def test_pi05_depth_align_supports_two_cameras():
    cfg = PI05DepthAlignConfig(enable=True, target_token_size=2, target_dim=3)
    depth_align_head = nn.Linear(5, 3)
    prefix_hidden_states = torch.randn(2, 4 + 4 + 8, 5)
    img_masks = [
        torch.tensor([True, True]),
        torch.tensor([True, False]),
    ]
    depth_targets = torch.randn(4, 4, 3)

    # 两个相机分别占 4 个图像 token，loss 只统计 img_masks 标记为 True 的相机。
    loss, depth_preds = compute_depth_alignment_loss(
        prefix_hidden_states=prefix_hidden_states,
        image_token_counts=[4, 4],
        img_masks=img_masks,
        depth_targets=depth_targets,
        depth_align_head=depth_align_head,
        config=cfg,
    )

    assert loss.ndim == 0
    assert depth_preds.shape == (4, 4, 3)


def test_depth_teacher_accepts_different_camera_aspect_ratios():
    class FakeMoge:
        def infer(self, images, **_):
            return {"depth": images.mean(dim=1)}

    class FakeLingBot:
        def infer_feat(self, images, depth, **_):
            values = images.mean(dim=(1, 2, 3))
            height = 2 if images.shape[-2] < images.shape[-1] else 3
            features = values[:, None, None, None].expand(-1, 3, height, 4).clone()
            return features, depth

    generator = PI05DepthTargetGenerator.__new__(PI05DepthTargetGenerator)
    generator.config = SimpleNamespace(resolution_level=3, target_num_tokens=16, target_token_size=2)
    generator.device = torch.device("cpu")
    generator.moge_model = FakeMoge()
    generator.lingbot_depth_model = FakeLingBot()
    front = torch.stack([torch.full((3, 6, 8), 0.1), torch.full((3, 6, 8), 0.2)])
    side = torch.stack([torch.full((3, 4, 8), 0.3), torch.full((3, 4, 8), 0.4)])

    targets = generator([front, side])

    assert targets.shape == (4, 4, 3)
    assert torch.allclose(targets[:, 0, 0], torch.tensor([0.1, 0.3, 0.2, 0.4]))


def test_pi05_depth_align_visualization_writes_debug_images(tmp_path):
    images = [torch.rand(2, 3, 224, 224) * 2 - 1, torch.rand(2, 3, 224, 224) * 2 - 1]
    img_masks = [torch.tensor([True, True]), torch.tensor([True, False])]
    depth_preds = torch.rand(4, 4, 3)
    depth_targets = torch.rand(4, 4, 3)

    # 可视化只用于人工 review，保存 RGB / target feature norm / pred feature norm 拼图。
    written = save_depth_alignment_visualization(
        images=images,
        depth_preds=depth_preds,
        depth_targets=depth_targets,
        img_masks=img_masks,
        output_dir=str(tmp_path),
        step=1,
        max_items=2,
        token_size=2,
    )

    assert len(written) == 2
    assert all(path.exists() for path in written)
