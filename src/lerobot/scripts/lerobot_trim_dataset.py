#!/usr/bin/env python

"""Analyze and trim long static prefixes from a LeRobot v3 dataset.

The source dataset is never modified. Rewriting creates a new dataset, resets
episode-relative timestamps, and preserves episode metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES

INTERNAL_EPISODE_COLUMNS = {
    "episode_index",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
    "meta/episodes/chunk_index",
    "meta/episodes/file_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Source LeRobot dataset root.")
    parser.add_argument("--repo-id", default=None, help="Repo id used to open the local source dataset.")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="New dataset root. Omit for analysis only."
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.01,
        help="A frame moves when max(abs(action - initial_action)) exceeds this value.",
    )
    parser.add_argument(
        "--min-moving-frames",
        type=int,
        default=5,
        help="Required consecutive moving frames before movement is accepted.",
    )
    parser.add_argument(
        "--keep-static-seconds",
        type=float,
        default=1.0,
        help="Static context retained immediately before the first sustained movement.",
    )
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--analysis-only", action="store_true")
    return parser.parse_args()


def find_first_sustained_motion(
    actions: np.ndarray,
    *,
    threshold: float,
    min_moving_frames: int,
) -> int:
    """Return the first frame of sustained displacement from the initial action."""
    if actions.ndim != 2 or len(actions) == 0:
        raise ValueError(f"Expected non-empty [T,D] actions, got {actions.shape}.")
    if threshold < 0:
        raise ValueError("motion_threshold must be non-negative.")
    if min_moving_frames <= 0:
        raise ValueError("min_moving_frames must be positive.")

    moving = np.max(np.abs(actions - actions[0]), axis=1) > threshold
    if len(moving) < min_moving_frames:
        return len(actions)
    sustained = np.convolve(
        moving.astype(np.int32),
        np.ones(min_moving_frames, dtype=np.int32),
        mode="valid",
    )
    starts = np.flatnonzero(sustained == min_moving_frames)
    return int(starts[0]) if len(starts) else len(actions)


def build_trim_plan(
    dataset: LeRobotDataset,
    *,
    motion_threshold: float,
    min_moving_frames: int,
    keep_static_seconds: float,
) -> pd.DataFrame:
    if keep_static_seconds < 0:
        raise ValueError("keep_static_seconds must be non-negative.")

    table = dataset.hf_dataset
    episode_ids = np.asarray(table["episode_index"], dtype=np.int64)
    actions = np.asarray(table["action"], dtype=np.float32)
    keep_frames = int(round(keep_static_seconds * dataset.fps))
    rows: list[dict[str, Any]] = []

    for episode_index in range(dataset.meta.total_episodes):
        positions = np.flatnonzero(episode_ids == episode_index)
        episode_actions = actions[positions]
        first_motion = find_first_sustained_motion(
            episode_actions,
            threshold=motion_threshold,
            min_moving_frames=min_moving_frames,
        )
        trim_frames = max(first_motion - keep_frames, 0)
        rows.append(
            {
                "episode_index": episode_index,
                "original_frames": len(positions),
                "first_sustained_motion_frame": first_motion,
                "initial_static_seconds": first_motion / dataset.fps,
                "trim_frames": trim_frames,
                "trim_seconds": trim_frames / dataset.fps,
                "output_frames": len(positions) - trim_frames,
            }
        )
    return pd.DataFrame(rows)


def write_analysis(
    plan: pd.DataFrame,
    *,
    report_dir: Path,
    source: Path,
    fps: int,
    motion_threshold: float,
    min_moving_frames: int,
    keep_static_seconds: float,
    destination: Path | None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(report_dir / "episode_trim_plan.csv", index=False)
    summary = {
        "source": str(source.resolve()),
        "fps": fps,
        "episodes": int(len(plan)),
        "original_frames": int(plan["original_frames"].sum()),
        "trimmed_frames": int(plan["trim_frames"].sum()),
        "output_frames": int(plan["output_frames"].sum()),
        "trimmed_seconds": float(plan["trim_seconds"].sum()),
        "median_initial_static_seconds": float(plan["initial_static_seconds"].median()),
        "mean_initial_static_seconds": float(plan["initial_static_seconds"].mean()),
        "max_initial_static_seconds": float(plan["initial_static_seconds"].max()),
        "episodes_over_5_seconds_static": int((plan["initial_static_seconds"] > 5.0).sum()),
        "motion_threshold": motion_threshold,
        "min_moving_frames": min_moving_frames,
        "keep_static_seconds": keep_static_seconds,
        "destination": str(destination.resolve()) if destination is not None else None,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info("Trim analysis: %s", json.dumps(summary, ensure_ascii=False))


def _task_lookup(dataset: LeRobotDataset) -> dict[int, str]:
    return {int(row.task_index): str(task) for task, row in dataset.meta.tasks.iterrows()}


def _extra_episode_metadata(episode: dict[str, Any], video_keys: list[str]) -> dict[str, Any]:
    video_prefixes = tuple(f"videos/{key}/" for key in video_keys)
    result = {}
    for key, value in episode.items():
        if key in INTERNAL_EPISODE_COLUMNS or key.startswith(video_prefixes):
            continue
        if hasattr(value, "item"):
            value = value.item()
        result[key] = value
    return result


def _restore_feature_shape(value: Any, feature: dict[str, Any]) -> Any:
    """Undo scalar squeezing performed by Hugging Face parquet decoding."""
    if feature["dtype"] == "string":
        return value
    expected_shape = tuple(feature["shape"])
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return array.reshape(expected_shape)


class EpisodeVideoReader:
    def __init__(self, dataset: LeRobotDataset, episode_index: int, trim_frames: int):
        self.captures: dict[str, cv2.VideoCapture] = {}
        episode = dataset.meta.episodes[episode_index]
        for key in dataset.meta.video_keys:
            path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
            capture = cv2.VideoCapture(str(path))
            start = int(round(float(episode[f"videos/{key}/from_timestamp"]) * dataset.fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start + trim_frames)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video: {path}")
            self.captures[key] = capture

    def read(self) -> dict[str, np.ndarray]:
        frames = {}
        for key, capture in self.captures.items():
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Video ended early while reading '{key}'.")
            frames[key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return frames

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()


def rewrite_dataset(
    source: LeRobotDataset,
    *,
    destination: Path,
    output_repo_id: str,
    plan: pd.DataFrame,
    vcodec: str,
    image_writer_threads: int,
) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    temporary = destination.with_name(f"{destination.name}.incomplete")
    if temporary.exists():
        raise FileExistsError(f"Incomplete destination already exists: {temporary}")

    features = {key: value for key, value in source.meta.features.items() if key not in DEFAULT_FEATURES}
    target = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=temporary,
        fps=source.fps,
        robot_type=source.meta.robot_type,
        features=features,
        use_videos=bool(source.meta.video_keys),
        image_writer_threads=image_writer_threads,
        vcodec=vcodec,
    )
    tasks = _task_lookup(source)
    episode_ids = np.asarray(source.hf_dataset["episode_index"], dtype=np.int64)

    try:
        for plan_row in plan.itertuples(index=False):
            episode_index = int(plan_row.episode_index)
            trim_frames = int(plan_row.trim_frames)
            positions = np.flatnonzero(episode_ids == episode_index)[trim_frames:]
            reader = EpisodeVideoReader(source, episode_index, trim_frames)
            try:
                for source_position in positions:
                    record = source.hf_dataset[int(source_position)]
                    frame = {
                        key: _restore_feature_shape(value, source.meta.features[key])
                        for key, value in record.items()
                        if key not in DEFAULT_FEATURES and key not in source.meta.video_keys
                    }
                    frame["task"] = tasks[int(record["task_index"])]
                    for key, rgb in reader.read().items():
                        frame[key] = rgb
                    target.add_frame(frame)
            finally:
                reader.close()

            episode = source.meta.episodes[episode_index]
            target.save_episode(
                parallel_encoding=False,
                extra_episode_metadata=_extra_episode_metadata(episode, source.meta.video_keys),
            )
            logging.info(
                "Saved episode %d/%d: trimmed=%d output=%d",
                episode_index + 1,
                source.meta.total_episodes,
                trim_frames,
                len(positions),
            )
        target.finalize()
        temporary.rename(destination)
    except Exception:
        target.stop_image_writer()
        logging.exception("Rewrite failed; partial output retained at %s", temporary)
        raise


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source_root = args.source.expanduser().resolve()
    repo_id = args.repo_id or f"local/{source_root.name}"
    dataset = LeRobotDataset(repo_id, root=source_root, download_videos=False)
    plan = build_trim_plan(
        dataset,
        motion_threshold=args.motion_threshold,
        min_moving_frames=args.min_moving_frames,
        keep_static_seconds=args.keep_static_seconds,
    )

    report_dir = args.report_dir
    if report_dir is None:
        report_dir = (
            args.output_dir.with_name(f"{args.output_dir.name}_trim_report")
            if args.output_dir is not None
            else Path("outputs/dataset_trim") / source_root.name
        )
    write_analysis(
        plan,
        report_dir=report_dir,
        source=source_root,
        fps=dataset.fps,
        motion_threshold=args.motion_threshold,
        min_moving_frames=args.min_moving_frames,
        keep_static_seconds=args.keep_static_seconds,
        destination=args.output_dir,
    )

    if args.analysis_only or args.output_dir is None:
        return
    output_repo_id = args.output_repo_id or f"{source_root.name}_trimmed"
    rewrite_dataset(
        dataset,
        destination=args.output_dir.expanduser().resolve(),
        output_repo_id=output_repo_id,
        plan=plan,
        vcodec=args.vcodec,
        image_writer_threads=args.image_writer_threads,
    )


if __name__ == "__main__":
    main()
