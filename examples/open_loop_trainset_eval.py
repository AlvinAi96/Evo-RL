#!/usr/bin/env python

"""Open-loop train-set evaluation for a saved LeRobot action policy.

The script samples frames from distinct training episodes, predicts complete action
chunks under several ACP prompt modes, compares them with the recorded action
chunks, and writes machine-readable metrics plus lightweight Pillow visualizations.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data._utils.collate import default_collate

# Importing the concrete config registers the draccus choice before train_config is decoded.
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.utils.constants import ACTION, OBS_STATE


DEFAULT_CHECKPOINT = Path(
    "outputs/train/pi05_insert_carrot_hil_acp_bs256_e10/checkpoints/001580/pretrained_model"
)
DEFAULT_OUTPUT = Path("outputs/open_loop/pi05_acp_epoch10_trainset_n12_seed2026")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cfg-beta", type=float, default=0.6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-start-frame", type=int, default=20)
    return parser.parse_args()


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        elif isinstance(value, list):
            cloned[key] = list(value)
        else:
            cloned[key] = value
    return cloned


def choose_samples(dataset: Any, indicator_field: str, count: int, seed: int, min_start: int) -> list[int]:
    if count < 2:
        raise ValueError("--num-samples must be at least 2 so both ACP classes can be represented.")

    table = dataset.hf_dataset
    indicators = np.asarray(table[indicator_field], dtype=np.int64)
    episode_ids = np.asarray(table["episode_index"], dtype=np.int64)
    frame_ids = np.asarray(table["frame_index"], dtype=np.int64)
    pad_margin = int(dataset.delta_indices[ACTION][-1]) if dataset.delta_indices else 0

    episode_lengths = {
        int(ep): int(length)
        for ep, length in zip(
            dataset.meta.episodes["episode_index"], dataset.meta.episodes["length"], strict=True
        )
    }
    valid = np.asarray(
        [
            frame >= min_start and frame + pad_margin < episode_lengths[int(ep)]
            for ep, frame in zip(episode_ids, frame_ids, strict=True)
        ],
        dtype=bool,
    )

    rng = np.random.default_rng(seed)
    target_counts = {1: count // 2, 0: count - count // 2}
    selected: list[int] = []
    used_episodes: set[int] = set()

    for label in (1, 0):
        candidates = np.flatnonzero(valid & (indicators == label))
        rng.shuffle(candidates)
        label_selected: list[int] = []

        # Prefer distinct episodes for visual diversity.
        for index in candidates:
            episode = int(episode_ids[index])
            if episode in used_episodes:
                continue
            label_selected.append(int(index))
            used_episodes.add(episode)
            if len(label_selected) == target_counts[label]:
                break

        # Fall back to repeated episodes only if a class is too concentrated.
        if len(label_selected) < target_counts[label]:
            already = set(label_selected)
            for index in candidates:
                if int(index) in already:
                    continue
                label_selected.append(int(index))
                if len(label_selected) == target_counts[label]:
                    break

        if len(label_selected) != target_counts[label]:
            raise RuntimeError(f"Could only select {len(label_selected)} samples for ACP label {label}.")
        selected.extend(label_selected)

    rng.shuffle(selected)
    return selected


def tensor_image_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(np.round(array * 255.0)), mode="RGB")


def save_observation_montage(sample: dict[str, Any], metadata: dict[str, Any], path: Path) -> None:
    keys = ["observation.images.front", "observation.images.side"]
    images = [tensor_image_to_pil(sample[key]) for key in keys if key in sample]
    thumb_w, thumb_h = 360, 270
    images = [image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS) for image in images]
    canvas = Image.new("RGB", (thumb_w * len(images), thumb_h + 54), "white")
    for column, image in enumerate(images):
        canvas.paste(image, (column * thumb_w, 54))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"sample {metadata['sample_id']} | dataset index {metadata['dataset_index']} | "
        f"episode {metadata['episode_index']} frame {metadata['frame_index']} | "
        f"ACP={metadata['acp_indicator']} ({metadata['episode_success']})"
    )
    draw.text((10, 8), title, fill="black")
    draw.text((10, 29), "front", fill="#333333")
    if len(images) > 1:
        draw.text((thumb_w + 10, 29), "side", fill="#333333")
    canvas.save(path, quality=92)


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    low: float,
    high: float,
    color: str,
    line_width: int,
) -> None:
    if high <= low:
        high = low + 1.0
    xs = np.linspace(x0, x0 + width, len(values))
    ys = y0 + height - (values - low) / (high - low) * height
    points = [(int(x), int(y)) for x, y in zip(xs, ys, strict=True)]
    if len(points) >= 2:
        draw.line(points, fill=color, width=line_width, joint="curve")


def save_action_plot(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    baseline: np.ndarray,
    action_names: list[str],
    title: str,
    path: Path,
) -> None:
    colors = {
        "ground truth": "#111111",
        "train_tag": "#2ca02c",
        "positive": "#d62728",
        "uncond": "#9467bd",
        "cfg_beta_0.6": "#1f77b4",
        "hold_state": "#999999",
    }
    width, row_height, left, right = 1120, 145, 125, 25
    top, bottom = 68, 24
    canvas = Image.new("RGB", (width, top + row_height * truth.shape[1] + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="black")
    legend_x = 12
    for label in ["ground truth", "train_tag", "positive", "uncond", "cfg_beta_0.6", "hold_state"]:
        draw.line((legend_x, 38, legend_x + 24, 38), fill=colors[label], width=4)
        draw.text((legend_x + 29, 30), label, fill="#222222")
        legend_x += 165

    plot_width = width - left - right
    for joint in range(truth.shape[1]):
        y = top + joint * row_height
        series = [truth[:, joint], baseline[:, joint]] + [value[:, joint] for value in predictions.values()]
        low = float(min(np.min(value) for value in series))
        high = float(max(np.max(value) for value in series))
        margin = max((high - low) * 0.08, 0.05)
        low -= margin
        high += margin
        draw.rectangle((left, y + 8, left + plot_width, y + row_height - 18), outline="#dddddd")
        for fraction in (0.25, 0.5, 0.75):
            gx = int(left + fraction * plot_width)
            draw.line((gx, y + 8, gx, y + row_height - 18), fill="#eeeeee")
        draw.text((8, y + 27), action_names[joint], fill="black")
        draw.text((8, y + 50), f"{low:.2f} .. {high:.2f}", fill="#666666")
        _draw_polyline(
            draw, baseline[:, joint], left, y + 8, plot_width, row_height - 26, low, high,
            colors["hold_state"], 2,
        )
        for mode, value in predictions.items():
            _draw_polyline(
                draw, value[:, joint], left, y + 8, plot_width, row_height - 26, low, high,
                colors[mode], 3,
            )
        _draw_polyline(
            draw, truth[:, joint], left, y + 8, plot_width, row_height - 26, low, high,
            colors["ground truth"], 4,
        )
    canvas.save(path)


def masked_values(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return array[np.broadcast_to(valid[..., None], array.shape)]


def correlation(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> float:
    x = masked_values(prediction, valid)
    y = masked_values(truth, valid)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def metric_block(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    selected = masked_values(error, valid)
    first_10_valid = valid.copy()
    first_10_valid[:, 10:] = False
    first_10_error = masked_values(error, first_10_valid)
    endpoint_error = error[:, -1]
    return {
        "mae": float(np.mean(np.abs(selected))),
        "rmse": float(np.sqrt(np.mean(np.square(selected)))),
        "first_step_mae": float(np.mean(np.abs(error[:, 0]))),
        "first_10_mae": float(np.mean(np.abs(first_10_error))),
        "endpoint_mae": float(np.mean(np.abs(endpoint_error))),
        "correlation": correlation(prediction, truth, valid),
    }


def compute_metrics(
    predictions: dict[str, np.ndarray],
    baseline: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    indicators: np.ndarray,
    action_names: list[str],
) -> dict[str, Any]:
    all_predictions = {**predictions, "hold_state": baseline}
    result: dict[str, Any] = {"overall": {}, "by_indicator": {}, "per_joint": {}}
    for mode, prediction in all_predictions.items():
        result["overall"][mode] = metric_block(prediction, truth, valid)
        result["by_indicator"][mode] = {}
        for label in (0, 1):
            mask = indicators == label
            result["by_indicator"][mode][str(label)] = metric_block(
                prediction[mask], truth[mask], valid[mask]
            )
        result["per_joint"][mode] = {}
        for joint, name in enumerate(action_names):
            joint_error = prediction[:, :, joint] - truth[:, :, joint]
            selected = joint_error[valid]
            x = prediction[:, :, joint][valid]
            y = truth[:, :, joint][valid]
            corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else float("nan")
            result["per_joint"][mode][name] = {
                "mae": float(np.mean(np.abs(selected))),
                "rmse": float(np.sqrt(np.mean(np.square(selected)))),
                "correlation": corr,
            }
    return result


def write_csv(metrics: dict[str, Any], path: Path) -> None:
    fieldnames = ["mode", "mae", "rmse", "first_step_mae", "first_10_mae", "endpoint_mae", "correlation"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode, values in metrics["overall"].items():
            writer.writerow({"mode": mode, **values})


def write_html(
    path: Path,
    metrics: dict[str, Any],
    sample_metadata: list[dict[str, Any]],
    cfg_beta: float,
) -> None:
    rows = []
    for mode, values in metrics["overall"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(mode)}</td>"
            f"<td>{values['mae']:.4f}</td><td>{values['rmse']:.4f}</td>"
            f"<td>{values['first_step_mae']:.4f}</td><td>{values['first_10_mae']:.4f}</td>"
            f"<td>{values['endpoint_mae']:.4f}</td><td>{values['correlation']:.4f}</td>"
            "</tr>"
        )
    cards = []
    for item in sample_metadata:
        sid = item["sample_id"]
        cards.append(
            "<section class='card'>"
            f"<h3>Sample {sid}: episode {item['episode_index']}, frame {item['frame_index']}, "
            f"ACP={item['acp_indicator']}</h3>"
            f"<img src='sample_{sid:02d}_observations.jpg' alt='observations {sid}'>"
            f"<img src='sample_{sid:02d}_actions.png' alt='action comparison {sid}'>"
            "</section>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pi0.5 ACP open-loop train-set evaluation</title>
<style>
body {{ font: 15px system-ui, sans-serif; margin: 28px; color: #222; }}
table {{ border-collapse: collapse; margin: 16px 0 28px; }}
th, td {{ border: 1px solid #ccc; padding: 7px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.card {{ border-top: 2px solid #ddd; padding-top: 12px; margin-top: 30px; }}
img {{ display: block; max-width: 1120px; width: 100%; margin: 10px 0; }}
code {{ background: #f3f3f3; padding: 2px 4px; }}
</style></head><body>
<h1>Pi0.5 ACP open-loop train-set evaluation</h1>
<p>All prompt modes use identical sampled frames and identical initial diffusion noise. The
<code>cfg_beta_{cfg_beta}</code> action is <code>uncond + {cfg_beta} × (positive − uncond)</code>.
Metrics compare predicted 50-step action chunks against recorded training actions in original
joint units. <code>hold_state</code> repeats the observed joint state as a simple baseline.</p>
<table><thead><tr><th>Mode</th><th>MAE</th><th>RMSE</th><th>Step-0 MAE</th>
<th>First-10 MAE</th><th>Endpoint MAE</th><th>Correlation</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
{''.join(cards)}
</body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable. Run this script on a GPU node.")

    print(f"Loading training config and dataset from {checkpoint}", flush=True)
    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    cfg.policy.device = args.device
    dataset = make_dataset(cfg)
    indicator_field = cfg.acp.indicator_field
    indices = choose_samples(dataset, indicator_field, args.num_samples, args.seed, args.min_start_frame)
    samples = [dataset[index] for index in indices]

    episode_success = {
        int(ep): status
        for ep, status in zip(
            dataset.meta.episodes["episode_index"], dataset.meta.episodes["episode_success"], strict=True
        )
    }
    sample_metadata: list[dict[str, Any]] = []
    for sample_id, (index, sample) in enumerate(zip(indices, samples, strict=True)):
        episode = int(sample["episode_index"])
        item = {
            "sample_id": sample_id,
            "dataset_index": index,
            "episode_index": episode,
            "frame_index": int(sample["frame_index"]),
            "acp_indicator": int(sample[indicator_field]),
            "advantage": float(sample["complementary_info.advantage_pistar06_run4_n50_r30"]),
            "episode_success": episode_success[episode],
        }
        sample_metadata.append(item)
        save_observation_montage(sample, item, output_dir / f"sample_{sample_id:02d}_observations.jpg")

    action_names = list(dataset.meta.features[ACTION].get("names") or [])
    if len(action_names) != int(samples[0][ACTION].shape[-1]):
        action_names = [f"action_{index}" for index in range(int(samples[0][ACTION].shape[-1]))]

    print(f"Selected dataset indices: {indices}", flush=True)
    print("Loading policy and saved processors", flush=True)
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(checkpoint, config=cfg.policy).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    truth = np.stack([sample[ACTION].numpy() for sample in samples])
    valid = ~np.stack([sample["action_is_pad"].numpy() for sample in samples])
    states = np.stack([sample[OBS_STATE].numpy() for sample in samples])
    baseline = np.repeat(states[:, None, :], truth.shape[1], axis=1)
    indicators = np.asarray([item["acp_indicator"] for item in sample_metadata], dtype=np.int64)
    base_tasks = [str(sample["task"]) for sample in samples]
    mode_tasks = {
        "train_tag": [
            build_acp_tagged_task(task, is_positive=bool(indicator))
            for task, indicator in zip(base_tasks, indicators, strict=True)
        ],
        "positive": [build_acp_tagged_task(task, is_positive=True) for task in base_tasks],
        "uncond": base_tasks,
    }

    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    all_noise = torch.randn(
        args.num_samples,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        generator=generator,
        device=args.device,
        dtype=torch.float32,
    )
    prediction_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    timings: dict[str, float] = defaultdict(float)

    with torch.inference_mode():
        for start in range(0, args.num_samples, args.batch_size):
            stop = min(start + args.batch_size, args.num_samples)
            raw_batch = default_collate(samples[start:stop])
            for mode in ("train_tag", "positive", "uncond"):
                batch = clone_batch(raw_batch)
                batch["task"] = mode_tasks[mode][start:stop]
                processed = preprocessor(batch)
                torch.cuda.synchronize()
                started = time.perf_counter()
                normalized = policy.predict_action_chunk(processed, noise=all_noise[start:stop].clone())
                prediction = postprocessor(normalized)
                torch.cuda.synchronize()
                timings[mode] += time.perf_counter() - started
                prediction_parts[mode].append(prediction.detach().float().cpu().numpy())
                print(f"Predicted {mode}: samples {start}:{stop}", flush=True)

    predictions = {mode: np.concatenate(parts) for mode, parts in prediction_parts.items()}
    cfg_mode = f"cfg_beta_{args.cfg_beta}"
    predictions[cfg_mode] = predictions["uncond"] + args.cfg_beta * (
        predictions["positive"] - predictions["uncond"]
    )
    if cfg_mode != "cfg_beta_0.6":
        # Plotting uses a stable display key while preserving the requested beta in saved arrays/metrics.
        predictions["cfg_beta_0.6"] = predictions.pop(cfg_mode)
        cfg_mode = "cfg_beta_0.6"

    metrics = compute_metrics(predictions, baseline, truth, valid, indicators, action_names)
    metrics["timings_s"] = dict(timings)
    metrics["config"] = {
        "checkpoint": str(checkpoint),
        "dataset_repo_id": cfg.dataset.repo_id,
        "dataset_root": str(cfg.dataset.root),
        "indicator_field": indicator_field,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "cfg_beta": args.cfg_beta,
        "indices": indices,
        "action_names": action_names,
    }
    metrics["samples"] = sample_metadata

    for sample_id, item in enumerate(sample_metadata):
        sample_predictions = {mode: value[sample_id] for mode, value in predictions.items()}
        title = (
            f"sample {sample_id} | ep {item['episode_index']} frame {item['frame_index']} | "
            f"ACP={item['acp_indicator']} | advantage={item['advantage']:.4f}"
        )
        save_action_plot(
            truth[sample_id], sample_predictions, baseline[sample_id], action_names, title,
            output_dir / f"sample_{sample_id:02d}_actions.png",
        )

    np.savez_compressed(
        output_dir / "predictions.npz",
        truth=truth,
        valid=valid,
        baseline_hold_state=baseline,
        indicators=indicators,
        indices=np.asarray(indices),
        **predictions,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    write_csv(metrics, output_dir / "summary.csv")
    write_html(output_dir / "report.html", metrics, sample_metadata, args.cfg_beta)

    print(json.dumps(metrics["overall"], indent=2, allow_nan=True), flush=True)
    print(f"Wrote report to {output_dir / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
