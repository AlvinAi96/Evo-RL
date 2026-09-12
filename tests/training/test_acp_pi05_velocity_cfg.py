#!/usr/bin/env python

from types import SimpleNamespace

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch


def test_pi05_velocity_cfg_guides_each_denoise_step():
    """用轻量桩模型确认 Pi0.5 每个 denoise step 都执行 velocity CFG。"""
    model = PI05Pytorch.__new__(PI05Pytorch)
    model.config = SimpleNamespace(num_inference_steps=2, chunk_size=1, max_action_dim=1)
    model.rtc_processor = None
    model._rtc_enabled = lambda: False
    model.sample_noise = lambda shape, device: torch.ones(shape, device=device)
    model.embed_prefix = lambda images, img_masks, tokens, masks: (
        torch.ones(tokens.shape[0], tokens.shape[1], 1, device=tokens.device),
        torch.ones(tokens.shape[0], tokens.shape[1], dtype=torch.bool, device=tokens.device),
        torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.bool, device=tokens.device),
    )
    model._prepare_attention_masks_4d = lambda masks: masks
    model._forward_calls = 0

    def fake_forward(**kwargs):
        """按调用顺序区分 cond/uncond prefix cache。"""
        model._forward_calls += 1
        cache_name = "cond" if model._forward_calls == 1 else "uncond"
        return [None, None], cache_name

    def fake_denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
        """cond velocity 固定为 3，uncond velocity 固定为 1，便于人工计算期望值。"""
        del prefix_pad_masks, timestep
        value = 3.0 if past_key_values == "cond" else 1.0
        return torch.full_like(x_t, value)

    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
        forward=fake_forward,
    )
    model.denoise_step = fake_denoise_step

    out = model.sample_actions_with_velocity_cfg(
        images=[],
        img_masks=[],
        cond_tokens=torch.ones(1, 2, dtype=torch.long),
        cond_masks=torch.ones(1, 2, dtype=torch.bool),
        uncond_tokens=torch.ones(1, 2, dtype=torch.long),
        uncond_masks=torch.ones(1, 2, dtype=torch.bool),
        cfg_beta=0.5,
    )

    assert torch.allclose(out, torch.full((1, 1, 1), -1.0))
