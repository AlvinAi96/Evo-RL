#!/usr/bin/env python

"""Reusable action-policy and Value-model diagnostics for LeRobot datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

RESERVED_ARRAYS = {"truth", "valid", "indices", "indicators", "baseline_hold_state"}


def _read_parquet_dataset(root: Path) -> pd.DataFrame:
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet data files found under {root}.")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _episode_lengths(root: Path) -> dict[int, int]:
    frames = []
    for path in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        frames.append(pd.read_parquet(path, columns=["episode_index", "length"]))
    if not frames:
        raise FileNotFoundError(f"No episode metadata found under {root}.")
    table = pd.concat(frames, ignore_index=True)
    return dict(zip(table["episode_index"].astype(int), table["length"].astype(int), strict=True))


def _save_heatmap(table: pd.DataFrame, row: str, column: str, value: str, path: Path) -> None:
    pivot = table.pivot(index=row, columns=column, values=value)
    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.quantile(finite, 0.95)) if len(finite) else 1.0
    vmax = max(vmax, 1e-9)
    cell_w, cell_h, left, top = 120, 54, 210, 70
    canvas = Image.new(
        "RGB",
        (left + cell_w * len(pivot.columns) + 20, top + cell_h * len(pivot.index) + 30),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), f"{value}: {row} x {column}", fill="black")
    for x, label in enumerate(pivot.columns):
        draw.text((left + x * cell_w + 6, 44), str(label), fill="black")
    for y, label in enumerate(pivot.index):
        draw.text((8, top + y * cell_h + 17), str(label), fill="black")
        for x in range(len(pivot.columns)):
            current = values[y, x]
            ratio = 0.0 if not np.isfinite(current) else min(max(current / vmax, 0.0), 1.0)
            color = (255, int(245 - 155 * ratio), int(220 - 190 * ratio))
            box = (
                left + x * cell_w,
                top + y * cell_h,
                left + (x + 1) * cell_w - 2,
                top + (y + 1) * cell_h - 2,
            )
            draw.rectangle(box, fill=color)
            text = "nan" if not np.isfinite(current) else f"{current:.3f}"
            draw.text((box[0] + 8, box[1] + 17), text, fill="black")
    canvas.save(path)


def analyze_actions(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.predictions)
    metadata = json.loads(args.summary.read_text(encoding="utf-8"))
    samples = pd.DataFrame(metadata["samples"])
    lengths = _episode_lengths(args.dataset_root)
    samples["episode_length"] = samples["episode_index"].map(lengths)
    samples["progress"] = samples["frame_index"] / np.maximum(samples["episode_length"] - 1, 1)
    samples["stage"] = np.minimum((samples["progress"] * args.stage_bins).astype(int), args.stage_bins - 1)

    truth = arrays["truth"]
    valid = arrays["valid"].astype(bool)
    action_names = metadata.get("config", {}).get("action_names")
    if not action_names or len(action_names) != truth.shape[-1]:
        action_names = [f"action_{index}" for index in range(truth.shape[-1])]
    modes = [key for key in arrays.files if key not in RESERVED_ARRAYS]
    horizons = sorted({min(value, truth.shape[1]) for value in args.horizons if value > 0})

    horizon_rows = []
    stage_rows = []
    for mode in modes:
        errors = np.abs(arrays[mode] - truth)
        for horizon in horizons:
            horizon_valid = valid[:, :horizon]
            for joint, name in enumerate(action_names):
                selected = errors[:, :horizon, joint][horizon_valid]
                horizon_rows.append(
                    {
                        "prompt": mode,
                        "horizon": horizon,
                        "seconds": horizon / args.fps,
                        "joint": name,
                        "mae": float(selected.mean()),
                        "rmse": float(np.sqrt(np.square(selected).mean())),
                    }
                )
        for stage in range(args.stage_bins):
            sample_mask = samples["stage"].to_numpy() == stage
            for joint, name in enumerate(action_names):
                selected = errors[sample_mask, :, joint][valid[sample_mask]]
                stage_rows.append(
                    {
                        "prompt": mode,
                        "stage": stage,
                        "progress_start": stage / args.stage_bins,
                        "progress_end": (stage + 1) / args.stage_bins,
                        "joint": name,
                        "windows": int(sample_mask.sum()),
                        "episodes": int(samples.loc[sample_mask, "episode_index"].nunique()),
                        "mae": float(selected.mean()) if len(selected) else float("nan"),
                    }
                )

    horizon_table = pd.DataFrame(horizon_rows)
    stage_table = pd.DataFrame(stage_rows)
    horizon_table.to_csv(output / "prompt_joint_horizon_mae.csv", index=False)
    stage_table.to_csv(output / "prompt_joint_stage_mae.csv", index=False)
    samples.to_csv(output / "sample_stages.csv", index=False)
    for mode in modes:
        _save_heatmap(
            horizon_table[horizon_table["prompt"] == mode],
            "joint",
            "horizon",
            "mae",
            output / f"{mode}_joint_horizon_mae.png",
        )
        _save_heatmap(
            stage_table[stage_table["prompt"] == mode],
            "joint",
            "stage",
            "mae",
            output / f"{mode}_joint_stage_mae.png",
        )
    print(f"Wrote action diagnostics to {output}")


def _value_metrics(root: Path, value_field: str, target_field: str, advantage_field: str) -> dict[str, Any]:
    table = _read_parquet_dataset(root)
    missing = {value_field, target_field, advantage_field} - set(table.columns)
    if missing:
        raise KeyError(f"{root} is missing inferred columns: {sorted(missing)}")
    values = np.asarray(table[value_field], dtype=np.float64)
    targets = np.asarray(table[target_field], dtype=np.float64)
    advantages = np.asarray(table[advantage_field], dtype=np.float64)
    residual = values - targets
    return {
        "dataset": str(root.resolve()),
        "frames": int(len(table)),
        "episodes": int(table["episode_index"].nunique()),
        "value_mae": float(np.mean(np.abs(residual))),
        "value_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "value_target_correlation": float(np.corrcoef(values, targets)[0, 1]),
        "advantage_mean": float(np.mean(advantages)),
        "advantage_std": float(np.std(advantages)),
        "advantage_p10": float(np.quantile(advantages, 0.1)),
        "advantage_p50": float(np.quantile(advantages, 0.5)),
        "advantage_p90": float(np.quantile(advantages, 0.9)),
    }


def analyze_value(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "train": _value_metrics(args.train_dataset, args.value_field, args.target_field, args.advantage_field)
    }
    if args.eval_dataset is not None:
        result["eval"] = _value_metrics(
            args.eval_dataset, args.value_field, args.target_field, args.advantage_field
        )
        result["generalization_gap_mae"] = result["eval"]["value_mae"] - result["train"]["value_mae"]
        result["overfit_warning"] = bool(
            result["eval"]["value_mae"] > result["train"]["value_mae"] * args.warning_ratio
        )
    else:
        result["overfit_warning"] = None
        result["limitation"] = "No held-out dataset supplied; training fit alone cannot diagnose overfitting."
    (output / "value_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame([value for value in result.values() if isinstance(value, dict)]).to_csv(
        output / "value_audit.csv", index=False
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    action = subparsers.add_parser("action", help="Analyze saved open-loop action predictions.")
    action.add_argument("--predictions", type=Path, required=True)
    action.add_argument("--summary", type=Path, required=True)
    action.add_argument("--dataset-root", type=Path, required=True)
    action.add_argument("--output-dir", type=Path, required=True)
    action.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 20, 30, 40, 50])
    action.add_argument("--stage-bins", type=int, default=5)
    action.add_argument("--fps", type=float, default=30.0)
    action.set_defaults(run=analyze_actions)

    value = subparsers.add_parser("value", help="Compare inferred Value against train/eval targets.")
    value.add_argument("--train-dataset", type=Path, required=True)
    value.add_argument("--eval-dataset", type=Path)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--value-field", default="complementary_info.value")
    value.add_argument("--target-field", default="complementary_info.value_target")
    value.add_argument("--advantage-field", default="complementary_info.advantage")
    value.add_argument("--warning-ratio", type=float, default=1.5)
    value.set_defaults(run=analyze_value)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
