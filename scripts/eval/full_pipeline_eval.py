#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
GAUSSIAN_ROOT = REPO_ROOT / "src" / "gaussiansplatting"
SCRIPT_ROOT = REPO_ROOT / "scripts" / "eval"
if str(GAUSSIAN_ROOT) not in sys.path:
    sys.path.insert(0, str(GAUSSIAN_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from iio_compat import load_array

PREFERRED_SUFFIXES = [".iio", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def shell_split(extra: str) -> List[str]:
    return shlex.split(extra) if extra.strip() else []


def run_timed(command: List[str], cwd: Path) -> Dict[str, object]:
    print(f"[RUN] cwd={cwd}")
    print("[RUN] " + " ".join(shlex.quote(part) for part in command))
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, check=True)
    elapsed = time.perf_counter() - start
    return {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
    }


def resolve_iteration(args: argparse.Namespace) -> int:
    if args.render_iteration is not None:
        return args.render_iteration
    return args.iterations


def opacity_label(opacity_threshold: Optional[float]) -> str:
    return str(opacity_threshold) if opacity_threshold is not None else "None"


def split_dir(model_path: Path, split: str, iteration: int, opacity_threshold: Optional[float]) -> Path:
    return model_path / f"{split}_op{opacity_label(opacity_threshold)}" / f"ours_{iteration}"


def choose_files_by_stem(directory: Path) -> Dict[str, Path]:
    grouped: Dict[str, List[Path]] = {}
    for path in directory.iterdir():
        if path.is_file():
            grouped.setdefault(path.stem, []).append(path)

    selected: Dict[str, Path] = {}
    for stem, paths in grouped.items():
        paths = sorted(paths, key=lambda path: PREFERRED_SUFFIXES.index(path.suffix.lower()) if path.suffix.lower() in PREFERRED_SUFFIXES else len(PREFERRED_SUFFIXES))
        selected[stem] = paths[0]
    return selected


def load_rgb_tensor(path: Path, device):
    import torch

    array = np.asarray(load_array(path), dtype=np.float32)
    if array.ndim == 2:
        array = array[..., None]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    array = np.clip(array, 0.0, 1.0)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_visual_metrics(render_root: Path, render_subdir: str, device) -> Dict[str, object]:
    import torch
    from lpipsPyTorch import lpips
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    renders_dir = render_root / render_subdir
    gt_dir = render_root / "gt"
    if not renders_dir.exists():
        raise FileNotFoundError(f"Render directory not found: {renders_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    render_files = choose_files_by_stem(renders_dir)
    gt_files = choose_files_by_stem(gt_dir)
    names = sorted(set(render_files).intersection(gt_files))
    if not names:
        raise RuntimeError(f"No paired render/GT files found in {renders_dir} and {gt_dir}")

    ssims: List[float] = []
    psnrs: List[float] = []
    lpipss: List[float] = []
    per_view: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for name in tqdm(names, desc="Visual metrics", leave=False):
            render = load_rgb_tensor(render_files[name], device)
            gt = load_rgb_tensor(gt_files[name], device)
            if render.shape != gt.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: render={tuple(render.shape)} gt={tuple(gt.shape)}"
                )

            ssim_val = float(ssim(render, gt).item())
            psnr_val = float(psnr(render, gt).item())
            lpips_val = float(lpips(render, gt, net_type="vgg").item())

            ssims.append(ssim_val)
            psnrs.append(psnr_val)
            lpipss.append(lpips_val)
            per_view[name] = {
                "SSIM": ssim_val,
                "PSNR": psnr_val,
                "LPIPS": lpips_val,
            }

    return {
        "split_root": str(render_root),
        "render_subdir": render_subdir,
        "num_views": len(names),
        "SSIM": float(np.mean(ssims)),
        "PSNR": float(np.mean(psnrs)),
        "LPIPS": float(np.mean(lpipss)),
        "per_view": per_view,
    }


def select_dsm_path(render_root: Path, explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        return explicit_path
    dsm_dir = render_root / "dsm"
    if not dsm_dir.exists():
        raise FileNotFoundError(f"DSM directory not found: {dsm_dir}")
    candidates = sorted(path for path in dsm_dir.iterdir() if path.is_file())
    if not candidates:
        raise RuntimeError(f"No DSM files found in {dsm_dir}")
    return candidates[-1]


def compute_dsm_metrics(
    render_root: Path,
    gt_dir: Path,
    aoi_id: str,
    out_dir: Path,
    explicit_dsm_path: Optional[Path],
    enable_vis_mask: bool,
    filter_tree: bool,
) -> Dict[str, object]:
    from eval_dsm import compute_mae_and_save_dsm_diff

    dsm_path = select_dsm_path(render_root, explicit_dsm_path)
    mae = compute_mae_and_save_dsm_diff(
        pred_dsm_path=str(dsm_path),
        gt_dir=str(gt_dir),
        aoi_id=aoi_id,
        out_dir=str(out_dir),
        enable_vis_mask=enable_vis_mask,
        filter_tree=filter_tree,
    )
    return {
        "pred_dsm_path": str(dsm_path),
        "gt_dir": str(gt_dir),
        "aoi_id": aoi_id,
        "MAE": float(mae),
    }


def build_train_command(args: argparse.Namespace, render_iteration: int) -> List[str]:
    command = [
        sys.executable,
        "train.py",
        "-s",
        str(args.source_path),
        "--images",
        str(args.images),
        "--eval",
        "-m",
        str(args.model_path),
        "--iterations",
        str(args.iterations),
    ]
    if render_iteration != args.iterations:
        command.extend(["--save_iterations", str(render_iteration)])
    command.extend(shell_split(args.train_extra_args))
    return command


def build_render_command(args: argparse.Namespace, render_iteration: int) -> List[str]:
    command = [
        sys.executable,
        "render.py",
        "-s",
        str(args.source_path),
        "--images",
        str(args.images),
        "-m",
        str(args.model_path),
        "--iteration",
        str(render_iteration),
        "--res",
        str(args.resolution),
    ]
    if args.render_split == "test":
        command.append("--skip_train")
    elif args.render_split == "train":
        command.append("--skip_test")
    if args.opacity_threshold is not None:
        command.extend(["--opacity_treshold", str(args.opacity_threshold)])
    command.extend(shell_split(args.render_extra_args))
    return command


def default_summary_path(model_path: Path, split: str, iteration: int) -> Path:
    return model_path / f"pipeline_eval_{split}_iter{iteration}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EOGS training, rendering, and evaluation end-to-end.")
    parser.add_argument("--source-path", required=True, type=Path, help="Affine scene directory used by train.py/render.py")
    parser.add_argument("--images", required=True, type=Path, help="Image directory passed to train.py/render.py")
    parser.add_argument("--model-path", required=True, type=Path, help="Output model directory")
    parser.add_argument("--gt-dir", type=Path, default=None, help="GT directory for DSM evaluation")
    parser.add_argument("--aoi-id", type=str, default=None, help="AOI id for DSM evaluation, e.g. JAX_068")
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations")
    parser.add_argument("--render-iteration", type=int, default=None, help="Checkpoint iteration to render/evaluate")
    parser.add_argument("--render-split", choices=["train", "test", "both"], default="train")
    parser.add_argument("--metric-split", choices=["train", "test"], default="train")
    parser.add_argument("--render-subdir", default="final", help="Rendered subdirectory for visual metrics")
    parser.add_argument("--opacity-threshold", type=float, default=None)
    parser.add_argument("--resolution", type=float, default=0.5, help="Render DSM resolution passed to render.py --res")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-rendering", action="store_true")
    parser.add_argument("--skip-visual-metrics", action="store_true")
    parser.add_argument("--skip-dsm", action="store_true")
    parser.add_argument("--train-extra-args", default="", help="Extra args appended to train.py")
    parser.add_argument("--render-extra-args", default="", help="Extra args appended to render.py")
    parser.add_argument("--dsm-path", type=Path, default=None, help="Optional explicit DSM path to evaluate")
    parser.add_argument("--dsm-out-dir", type=Path, default=None, help="Where to save rDSM/rDSM diff outputs")
    parser.add_argument("--disable-vis-mask", action="store_true", help="Disable DSM visibility mask")
    parser.add_argument("--disable-tree-filter", action="store_true", help="Disable DSM tree filtering")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_iteration = resolve_iteration(args)
    args.model_path.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "source_path": str(args.source_path),
        "images": str(args.images),
        "model_path": str(args.model_path),
        "iterations": args.iterations,
        "render_iteration": render_iteration,
        "render_split": args.render_split,
        "metric_split": args.metric_split,
        "render_subdir": args.render_subdir,
        "timing_seconds": {},
        "commands": {},
    }

    if not args.skip_training:
        train_result = run_timed(build_train_command(args, render_iteration), cwd=GAUSSIAN_ROOT)
        summary["commands"]["train"] = train_result["command"]
        summary["timing_seconds"]["train"] = train_result["elapsed_seconds"]
    else:
        summary["timing_seconds"]["train"] = None

    if not args.skip_rendering:
        render_result = run_timed(build_render_command(args, render_iteration), cwd=GAUSSIAN_ROOT)
        summary["commands"]["render"] = render_result["command"]
        summary["timing_seconds"]["render"] = render_result["elapsed_seconds"]
    else:
        summary["timing_seconds"]["render"] = None

    metric_root = split_dir(args.model_path, args.metric_split, render_iteration, args.opacity_threshold)
    summary["metric_root"] = str(metric_root)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary["metric_device"] = str(device)

    if not args.skip_visual_metrics:
        visual_start = time.perf_counter()
        summary["visual_metrics"] = compute_visual_metrics(metric_root, args.render_subdir, device)
        summary["timing_seconds"]["visual_metrics"] = time.perf_counter() - visual_start
    else:
        summary["timing_seconds"]["visual_metrics"] = None

    if not args.skip_dsm:
        if args.gt_dir is None or args.aoi_id is None:
            raise ValueError("--gt-dir and --aoi-id are required unless --skip-dsm is set")
        dsm_out_dir = args.dsm_out_dir or args.model_path
        dsm_out_dir.mkdir(parents=True, exist_ok=True)
        dsm_start = time.perf_counter()
        summary["dsm_metrics"] = compute_dsm_metrics(
            render_root=metric_root,
            gt_dir=args.gt_dir,
            aoi_id=args.aoi_id,
            out_dir=dsm_out_dir,
            explicit_dsm_path=args.dsm_path,
            enable_vis_mask=not args.disable_vis_mask,
            filter_tree=not args.disable_tree_filter,
        )
        summary["timing_seconds"]["dsm_metrics"] = time.perf_counter() - dsm_start
    else:
        summary["timing_seconds"]["dsm_metrics"] = None

    summary_path = args.summary_path or default_summary_path(args.model_path, args.metric_split, render_iteration)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== Pipeline Summary ===")
    print(f"Summary JSON: {summary_path}")
    if "visual_metrics" in summary:
        print(
            "Visual metrics: "
            f"SSIM={summary['visual_metrics']['SSIM']:.4f}, "
            f"PSNR={summary['visual_metrics']['PSNR']:.4f}, "
            f"LPIPS={summary['visual_metrics']['LPIPS']:.4f}"
        )
    if "dsm_metrics" in summary:
        print(f"DSM MAE: {summary['dsm_metrics']['MAE']:.4f}")
    print(
        "Timing (s): "
        f"train={summary['timing_seconds']['train']}, "
        f"render={summary['timing_seconds']['render']}, "
        f"visual={summary['timing_seconds']['visual_metrics']}, "
        f"dsm={summary['timing_seconds']['dsm_metrics']}"
    )


if __name__ == "__main__":
    main()
