---
library_name: lerobot
tags:
  - lerobot
  - robotics
  - vision-language-action
  - pi05
  - advantage-conditioned-policy
  - human-in-the-loop
datasets:
  - AlvinAi/insert_carrot_into_the_hole_hil
---

# Pi0.5 ACP Policy — Insert Carrot into the Hole

This repository contains a **Pi0.5 action policy** fine-tuned for the task **"insert carrot into the hole"** with the Advantage-Conditioned Policy (ACP) stage of the [MINT-SJTU/Evo-RL](https://github.com/MINT-SJTU/Evo-RL) pipeline.

Despite `pistar06` appearing in the repository name, this is not a Pi*0.6 value model. The action-policy architecture saved here is `pi05`; Pi*0.6 was the value model used to derive the advantage/ACP labels for policy training.

## Checkpoints

The default `main` revision contains the final epoch-10 policy. Selected checkpoints are also available as directly downloadable branches:

| Revision | Optimizer step | Approximate dataset passes |
|---|---:|---:|
| [`epoch-03`](https://huggingface.co/Aikwed/pistar06_insert_carrot_into_the_hole_acp_r1/tree/epoch-03) | 474 | 3.01 |
| [`epoch-05`](https://huggingface.co/Aikwed/pistar06_insert_carrot_into_the_hole_acp_r1/tree/epoch-05) | 790 | 5.01 |
| [`epoch-07`](https://huggingface.co/Aikwed/pistar06_insert_carrot_into_the_hole_acp_r1/tree/epoch-07) | 1,106 | 7.02 |
| [`main`](https://huggingface.co/Aikwed/pistar06_insert_carrot_into_the_hole_acp_r1) / [`epoch-10`](https://huggingface.co/Aikwed/pistar06_insert_carrot_into_the_hole_acp_r1/tree/epoch-10) | 1,580 | 10.02 |

`main` and `epoch-10` point to the same checkpoint.

## Training data

- Dataset: [AlvinAi/insert_carrot_into_the_hole_hil](https://huggingface.co/datasets/AlvinAi/insert_carrot_into_the_hole_hil)
- Frames: 40,353
- Episodes: 83
- Task text: `insert carrot into the hole`
- Cameras: `observation.images.front` and `observation.images.side`
- Robot state: 6 joint-position dimensions
- Action: 6 joint-position dimensions
- ACP indicator: `complementary_info.acp_indicator_pistar06_run4_n50_r30`
- Positive ACP-label ratio: approximately 30%

During preprocessing, `front` is mapped to `base_0_rgb` and `side` is mapped to `left_wrist_0_rgb`. The unused `right_wrist_0_rgb` input is represented as a masked, padded camera by the Pi0.5 implementation.

## Model and training configuration

| Parameter | Value |
|---|---:|
| Policy type | `pi05` |
| Initialization | `lerobot/pi05_base` |
| Parameters | 3.62B |
| Training dtype | bfloat16 |
| GPUs | 4 |
| Batch size per GPU | 64 |
| Global batch size | 256 |
| Optimizer steps | 1,580 |
| Dataset passes | 10.02 |
| Optimizer | AdamW |
| Peak learning rate | 2.5e-5 |
| Final learning rate | 2.5e-6 |
| Gradient clipping | 1.0 |
| Gradient checkpointing | enabled |
| Action chunk size | 50 |
| Inference denoising steps | 10 |
| ACP indicator dropout | 0.3 |

All model parameters, including the vision-language backbone and action expert, were fine-tuned. ACP indicator dropout teaches both tagged and untagged task conditions.

The recorded training loss decreased from approximately `0.332` near the start to `0.028` at the final step. These are training metrics, not held-out evaluation or closed-loop robot success rates.

## Loading

Use the compatible Evo-RL/LeRobot checkout and load the default final checkpoint with:

```python
from lerobot.policies.pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline

repo_id = "Aikwed/pistar06_insert_carrot_into_the_hole_acp_r1"

policy = PI05Policy.from_pretrained(repo_id)
preprocessor = PolicyProcessorPipeline.from_pretrained(
    repo_id,
    config_filename="policy_preprocessor.json",
)
postprocessor = PolicyProcessorPipeline.from_pretrained(
    repo_id,
    config_filename="policy_postprocessor.json",
)
```

To load a selected checkpoint robustly across code versions, download its complete revision first and then load the local snapshot:

```python
from huggingface_hub import snapshot_download
from lerobot.policies.pi05 import PI05Policy
from lerobot.processor import PolicyProcessorPipeline

model_dir = snapshot_download(
    repo_id="Aikwed/pistar06_insert_carrot_into_the_hole_acp_r1",
    revision="epoch-07",  # epoch-03, epoch-05, epoch-07, or epoch-10
)

policy = PI05Policy.from_pretrained(model_dir)
preprocessor = PolicyProcessorPipeline.from_pretrained(
    model_dir,
    config_filename="policy_preprocessor.json",
)
postprocessor = PolicyProcessorPipeline.from_pretrained(
    model_dir,
    config_filename="policy_postprocessor.json",
)
```

The saved preprocessor references Evo-RL's `relative_actions_processor` registry step (disabled for this absolute-action checkpoint). Use a checkout that includes this processor registration when restoring the complete preprocessing pipeline.

## Included artifacts

- Pi0.5 model weights in `model.safetensors`
- Policy and training configurations
- Preprocessor and postprocessor definitions
- State/action normalization and unnormalization statistics

Optimizer, scheduler, and RNG states are intentionally not included; these checkpoints are intended for inference and evaluation rather than exact training resumption.

## Limitations

- No held-out or closed-loop robot evaluation is included yet, so the best epoch has not been established.
- Low offline training loss does not by itself demonstrate improved task success.
- The policy is task-, camera-layout-, and embodiment-specific.
- Observation names, state/action ordering, task prompting, normalization, and preprocessing must match the saved configuration.
- Real-robot use requires task-specific safety limits and supervised validation.
