#!/usr/bin/env python

import torch
from torch import nn

from lerobot.policies.pi05.configuration_pi05 import PI05Config, PI05DepthAlignConfig
from lerobot.policies.pi05.depth_align import (
    compute_depth_alignment_loss,
    save_depth_alignment_visualization,
)


def test_pi05_depth_align_defaults_to_disabled():
    cfg = PI05Config(max_action_dim=7, max_state_dim=14, dtype="float32")

    assert cfg.depth_align.enable is False
    assert cfg.depth_align.visualize.enable is False


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
