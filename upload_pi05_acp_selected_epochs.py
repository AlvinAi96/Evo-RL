#!/usr/bin/env python

from pathlib import Path

from huggingface_hub import HfApi


OUTPUT_DIR = Path(
    "/scratch/u6pw/ql337.u6pw/projects/Evo-RL/outputs/train/"
    "pi05_insert_carrot_hil_acp_bs256_e10"
)
REPO_ID = "Aikwed/pistar06_insert_carrot_into_the_hole_acp_r1"
STEPS_PER_EPOCH = 158
EPOCHS_TO_UPLOAD = (3, 5, 7, 10)


def checkpoint_model_dir(epoch: int) -> Path:
    step = epoch * STEPS_PER_EPOCH
    return OUTPUT_DIR / "checkpoints" / f"{step:06d}" / "pretrained_model"


def main() -> None:
    model_dirs = {epoch: checkpoint_model_dir(epoch) for epoch in EPOCHS_TO_UPLOAD}
    for epoch, model_dir in model_dirs.items():
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Epoch {epoch} checkpoint is incomplete: {model_dir}")
        if (model_dir.parent / "training_state").exists():
            raise RuntimeError(f"Unexpected training state in epoch {epoch}: {model_dir.parent}")

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # Keep epoch 10 on main so the default policy path resolves to the final model.
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="model",
        folder_path=model_dirs[10],
        revision="main",
        commit_message="Upload ACP Pi0.5 epoch 10",
    )

    # Preserve the requested checkpoints as directly loadable Hub revisions.
    for epoch in EPOCHS_TO_UPLOAD:
        revision = f"epoch-{epoch:02d}"
        api.create_branch(repo_id=REPO_ID, repo_type="model", branch=revision, exist_ok=True)
        if epoch != 10:
            api.upload_folder(
                repo_id=REPO_ID,
                repo_type="model",
                folder_path=model_dirs[epoch],
                revision=revision,
                commit_message=f"Upload ACP Pi0.5 epoch {epoch}",
            )
        print(f"Uploaded epoch {epoch}: https://huggingface.co/{REPO_ID}/tree/{revision}")


if __name__ == "__main__":
    main()
