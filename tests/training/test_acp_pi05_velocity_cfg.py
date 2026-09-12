#!/usr/bin/env python

from types import SimpleNamespace

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def test_pi05_velocity_cfg_guides_each_denoise_step():
    """用轻量桩模型确认 Pi0.5 sample_actions 每个 denoise step 都执行 velocity CFG。"""
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
        """按调用顺序区分 base/positive prefix cache。"""
        model._forward_calls += 1
        cache_name = "base" if model._forward_calls == 1 else "positive"
        return [None, None], cache_name

    def fake_denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
        """positive velocity 固定为 3，base velocity 固定为 1，便于人工计算期望值。"""
        del prefix_pad_masks, timestep
        value = 3.0 if past_key_values == "positive" else 1.0
        return torch.full_like(x_t, value)

    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
        forward=fake_forward,
    )
    model.denoise_step = fake_denoise_step

    out = model.sample_actions(
        images=[],
        img_masks=[],
        tokens=torch.ones(1, 2, dtype=torch.long),
        masks=torch.ones(1, 2, dtype=torch.bool),
        cfg_positive_tokens=torch.ones(1, 2, dtype=torch.long),
        cfg_positive_masks=torch.ones(1, 2, dtype=torch.bool),
        cfg_beta=0.5,
    )

    assert torch.allclose(out, torch.full((1, 1, 1), -1.0))


def test_pi05_policy_predict_action_chunk_passes_positive_tokens_before_postprocess():
    """确认 policy wrapper 只把 positive tokens 传入 sample_actions，不在外层做动作相减。"""
    policy = PI05Policy.__new__(PI05Policy)
    policy.config = SimpleNamespace(output_features={ACTION: SimpleNamespace(shape=(1,))})
    policy.eval = lambda: None
    policy._preprocess_images = lambda batch: (["image"], ["mask"])
    calls = []

    class _DummyModel:
        def sample_actions(self, images, img_masks, tokens, masks, **kwargs):
            """记录传入 sample_actions 的 CFG tokens，模拟已引导的归一化 action chunk。"""
            calls.append(
                {
                    "images": images,
                    "tokens": tokens,
                    "masks": masks,
                    "cfg_positive_tokens": kwargs["cfg_positive_tokens"],
                    "cfg_positive_masks": kwargs["cfg_positive_masks"],
                    "cfg_beta": kwargs["cfg_beta"],
                }
            )
            return torch.tensor([[[7.0]]], dtype=torch.float32)

    policy.model = _DummyModel()
    base_batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2]], dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, True]]),
    }
    positive_batch = {
        OBS_LANGUAGE_TOKENS: torch.tensor([[3, 4]], dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor([[True, False]]),
    }

    out = policy.predict_action_chunk(base_batch, cfg_positive_batch=positive_batch, cfg_beta=0.8)

    assert torch.equal(out, torch.tensor([[[7.0]]], dtype=torch.float32))
    assert torch.equal(calls[0]["tokens"], base_batch[OBS_LANGUAGE_TOKENS])
    assert torch.equal(calls[0]["cfg_positive_tokens"], positive_batch[OBS_LANGUAGE_TOKENS])
    assert calls[0]["cfg_beta"] == 0.8
