#!/usr/bin/env python

from pathlib import Path

import torch

from lerobot.policies.pi05.configuration_pi05 import PI05DepthAlignConfig
from lerobot.policies.pi05.depth_align import PI05DepthTargetGenerator


MOGE_PATH = Path("/lus/lfs1aip2/projects/u6pw/models/evo_rl_depth/moge-2-vitb-normal/model.pt")
LINGBOT_DEPTH_PATH = Path(
    "/lus/lfs1aip2/projects/u6pw/models/evo_rl_depth/lingbot-depth-pretrain-vitl-14/model.pt"
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the depth dependency smoke test.")

    config = PI05DepthAlignConfig(
        enable=True,
        moge_path=str(MOGE_PATH),
        lingbot_depth_path=str(LINGBOT_DEPTH_PATH),
        target_token_size=16,
        target_dim=1024,
        target_num_tokens=256,
        resolution_level=3,
    )
    generator = PI05DepthTargetGenerator(config, torch.device("cuda"))
    images = [torch.rand(1, 3, 224, 224, device="cuda") * 2 - 1 for _ in range(2)]
    targets = generator(images)
    expected_shape = (2, 256, 1024)
    if tuple(targets.shape) != expected_shape:
        raise RuntimeError(f"Unexpected depth target shape: {tuple(targets.shape)}, expected {expected_shape}")
    if not bool(torch.isfinite(targets).all()):
        raise RuntimeError("Depth targets contain non-finite values.")
    print(f"depth smoke test passed: shape={tuple(targets.shape)} dtype={targets.dtype}")


if __name__ == "__main__":
    main()
