#!/usr/bin/env python

"""Compare MoGe depth inference on native-aspect and Pi0.5 letterboxed RGB."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--moge-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--moge-source",
        type=Path,
        default=None,
        help="Directory containing the 'moge' package when it is not installed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["0:0.25:front", "0:0.75:side"],
        help="episode:progress:camera entries.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution-level", type=int, default=3)
    parser.add_argument("--native-num-tokens", type=int, default=1200)
    parser.add_argument("--policy-num-tokens", type=int, default=256)
    parser.add_argument("--policy-size", type=int, default=224)
    return parser.parse_args()


def _episodes(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True).set_index("episode_index")


def _read_frame(
    root: Path,
    episode: pd.Series,
    episode_index: int,
    frame: int,
    camera: str,
    dataset_fps: float,
) -> np.ndarray:
    key = f"observation.images.{camera}"
    prefix = f"videos/{key}"
    path = (
        root
        / prefix
        / f"chunk-{int(episode[prefix + '/chunk_index']):03d}"
        / f"file-{int(episode[prefix + '/file_index']):03d}.mp4"
    )
    capture = cv2.VideoCapture(str(path))
    absolute_frame = round(
        (float(episode[prefix + "/from_timestamp"]) + frame / dataset_fps)
        * float(capture.get(cv2.CAP_PROP_FPS))
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode episode={episode_index} frame={frame} camera={camera}.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _letterbox(image: torch.Tensor, size: int) -> tuple[torch.Tensor, np.ndarray]:
    height, width = image.shape[-2:]
    ratio = max(height / size, width / size)
    resized_h, resized_w = int(height / ratio), int(width / ratio)
    resized = F.interpolate(image[None], size=(resized_h, resized_w), mode="bilinear", align_corners=False)[0]
    top = (size - resized_h) // 2
    bottom = size - resized_h - top
    left = (size - resized_w) // 2
    right = size - resized_w - left
    padded = F.pad(resized, (left, right, top, bottom), value=0.0)
    mask = np.zeros((size, size), dtype=bool)
    mask[top : size - bottom if bottom else size, left : size - right if right else size] = True
    return padded, mask


def _depth_image(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((depth - low) / max(high - low, 1e-8), 0.0, 1.0)
    gray = np.uint8(np.round((1.0 - normalized) * 255.0))
    return cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def _save_montage(
    rgb: np.ndarray, padded: np.ndarray, native: np.ndarray, square: np.ndarray, path: Path
) -> None:
    low, high = np.nanquantile(native, [0.02, 0.98])
    panels = [
        cv2.resize(rgb, (448, 336), interpolation=cv2.INTER_AREA),
        cv2.resize(np.uint8(np.round(padded * 255)), (448, 336), interpolation=cv2.INTER_NEAREST),
        cv2.resize(_depth_image(native, low, high), (448, 336), interpolation=cv2.INTER_NEAREST),
        cv2.resize(_depth_image(square, low, high), (448, 336), interpolation=cv2.INTER_NEAREST),
    ]
    canvas = np.concatenate([np.concatenate(panels[:2], axis=1), np.concatenate(panels[2:], axis=1)], axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    if args.moge_source is not None:
        sys.path.insert(0, str(args.moge_source.resolve()))
    from moge.model.v2 import MoGeModel

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.moge_checkpoint, map_location="cpu", weights_only=True)
    model = MoGeModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(args.device).eval()
    episodes = _episodes(args.dataset_root)
    dataset_fps = float(
        json.loads((args.dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))["fps"]
    )
    rows = []

    for spec in args.samples:
        episode_index_text, progress_text, camera = spec.split(":", maxsplit=2)
        episode_index, progress = int(episode_index_text), float(progress_text)
        episode = episodes.loc[episode_index]
        frame = int(round((int(episode["length"]) - 1) * progress))
        rgb = _read_frame(args.dataset_root, episode, episode_index, frame, camera, dataset_fps)
        native_input = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(args.device) / 255.0
        padded_input, content_mask = _letterbox(native_input, args.policy_size)
        started = time.monotonic()
        with torch.inference_mode():
            native_result = model.infer(
                native_input,
                num_tokens=args.native_num_tokens,
                resolution_level=args.resolution_level,
                apply_mask=False,
            )
            padded_result = model.infer(
                padded_input,
                num_tokens=args.policy_num_tokens,
                resolution_level=args.resolution_level,
                apply_mask=False,
            )
        native_depth = native_result["depth"].detach().float().cpu().numpy()
        padded_depth = padded_result["depth"].detach().float().cpu().numpy()
        stem = f"ep{episode_index:03d}_f{frame:05d}_{camera}"
        np.savez_compressed(
            output / f"{stem}.npz",
            rgb=rgb,
            padded_rgb=padded_input.detach().cpu().numpy(),
            native_depth=native_depth,
            padded_depth=padded_depth,
            content_mask=content_mask,
        )
        _save_montage(
            rgb,
            padded_input.detach().cpu().permute(1, 2, 0).numpy(),
            native_depth,
            padded_depth,
            output / f"{stem}.png",
        )
        rows.append(
            {
                "episode": episode_index,
                "frame": frame,
                "progress": progress,
                "camera": camera,
                "native_shape": list(native_input.shape),
                "padding_fraction": float(1.0 - content_mask.mean()),
                "native_depth_p02": float(np.nanquantile(native_depth, 0.02)),
                "native_depth_p98": float(np.nanquantile(native_depth, 0.98)),
                "padded_depth_p02": float(np.nanquantile(padded_depth, 0.02)),
                "padded_depth_p98": float(np.nanquantile(padded_depth, 0.98)),
                "seconds": time.monotonic() - started,
            }
        )
        print(rows[-1], flush=True)

    (output / "depth_audit.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "depth_audit.csv", index=False)


if __name__ == "__main__":
    main()
