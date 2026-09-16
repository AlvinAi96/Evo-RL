#!/usr/bin/env python

import json
from pathlib import Path

from huggingface_hub import HfApi


RUN_ROOT = Path(
    "/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/train/"
    "pi05_insert_carrot_v2_r1_acp_depth_bs256_e10"
)
CHECKPOINT_LINK = RUN_ROOT / "checkpoints" / "last"
EXPECTED_FINAL_STEP = 1687
REPO_ID = "Aikwed/pi05_insert_carrot_v2_r1_acp_depth_n50_r30"


def main() -> None:
    checkpoint_dir = CHECKPOINT_LINK.resolve(strict=True)
    pretrained_dir = checkpoint_dir / "pretrained_model"
    training_step_path = checkpoint_dir / "training_state" / "training_step.json"
    model_path = pretrained_dir / "model.safetensors"

    for path in (training_step_path, model_path, pretrained_dir / "train_config.json"):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Final checkpoint file is missing or empty: {path}")

    step = int(json.loads(training_step_path.read_text())["step"])
    if step != EXPECTED_FINAL_STEP:
        raise RuntimeError(f"Expected final step {EXPECTED_FINAL_STEP}, found {step} in {checkpoint_dir}")

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=pretrained_dir,
        revision="main",
        commit_message="Upload Pi0.5 ACP depth-alignment epoch 10",
    )

    revision = "epoch-10"
    api.create_branch(repo_id=REPO_ID, repo_type="model", branch=revision, exist_ok=True)
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=pretrained_dir,
        revision=revision,
        commit_message="Upload Pi0.5 ACP depth-alignment epoch 10",
    )
    print(f"Uploaded final model: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
