import argparse
import json
import os
from glob import glob
import numpy as np
from plyfile import PlyData


def read_affine(path: str):
    with open(path, "r") as f:
        metas = json.load(f)
    if len(metas) == 0:
        raise RuntimeError(f"No metadata in {path}")
    model = metas[0]["model"]
    camera_models = []
    for m in metas:
        if "model" not in m or "coef_" not in m["model"] or "intercept_" not in m["model"]:
            continue
        A = np.asarray(m["model"]["coef_"], dtype=np.float64)
        b = np.asarray(m["model"]["intercept_"], dtype=np.float64)
        altitude_bounds = np.asarray([m["min_alt"], m["max_alt"]], dtype=np.float64)
        camera_models.append((A, b, altitude_bounds))
    return (
        np.asarray(model["center"], dtype=np.float64),
        float(model["scale"]),
        np.asarray(model["min_world"], dtype=np.float64),
        np.asarray(model["max_world"], dtype=np.float64),
        camera_models,
    )


def read_ply_xyz(path: str):   
    if not os.path.exists(path):
        parent = os.path.dirname(path) or "."
        basename = os.path.basename(path)
        suggestions = []
        if os.path.isdir(parent):
            for cand in glob(os.path.join(parent, "*.ply")):
                cand_base = os.path.basename(cand)
                if cand_base.lower() == basename.lower() or "points3d" in cand_base.lower():
                    suggestions.append(cand)
        suggestion_msg = ""
        if suggestions:
            suggestion_msg = "\nDid you mean one of:\n  - " + "\n  - ".join(sorted(set(suggestions)))
        raise FileNotFoundError(f"Input ply not found: {path}{suggestion_msg}")
    ply = PlyData.read(path)
    v = ply["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)


def ratio_in_bounds(pts: np.ndarray, low: np.ndarray, high: np.ndarray):
    return float(np.mean(np.all((pts >= low[None]) & (pts <= high[None]), axis=1)))


def summarize(name: str, pts: np.ndarray, low: np.ndarray, high: np.ndarray):
    print(f"\n[{name}]")
    print(f"points: {len(pts)}")
    print(f"min: {pts.min(axis=0)}")
    print(f"max: {pts.max(axis=0)}")
    print(f"in-bounds ratio: {ratio_in_bounds(pts, low, high):.4f}")

def camera_consistency_ratio(pts_world: np.ndarray, A: np.ndarray, b: np.ndarray, altitude_bounds: np.ndarray):
    uva = pts_world @ A.T + b[None]
    uv_ok = np.all(np.abs(uva[:, :2]) <= 1.0, axis=1)
    alt_ok = (uva[:, 2] >= altitude_bounds[0]) & (uva[:, 2] <= altitude_bounds[1])
    return float(np.mean(uv_ok & alt_ok))

def camera_consistency_stats(pts_world: np.ndarray, camera_models):
    if len(camera_models) == 0:
        return float("nan"), float("nan"), float("nan")
    ratios = [camera_consistency_ratio(pts_world, A, b, altitude_bounds) for A, b, altitude_bounds in camera_models]
    ratios = np.asarray(ratios, dtype=np.float64)
    return float(ratios.min()), float(ratios.mean()), float(ratios.max())


def main():
    parser = argparse.ArgumentParser(
        description="Check whether a point cloud aligns with affine_models normalization."
    )
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--affine-models-json", required=True)
    parser.add_argument("--bounds-margin", type=float, default=0.10)
    parser.add_argument(
        "--input-coord",
        choices=["auto", "normalized", "utm", "bbox_fit"],
        default="auto",
        help=(
            "How to interpret --input-ply for diagnostics. "
            "'bbox_fit' is treated as 'normalized' because converter outputs world coordinates."
        ),
    )
    args = parser.parse_args()

    center, scale, min_world, max_world, camera_models = read_affine(args.affine_models_json)
    xyz = read_ply_xyz(args.input_ply)

    extent = max_world - min_world
    low = min_world - args.bounds_margin * extent
    high = max_world + args.bounds_margin * extent

    xyz_as_normalized = xyz
    xyz_as_utm = (xyz - center[None]) / scale


    if args.input_coord in ("normalized", "bbox_fit"):
        if args.input_coord == "bbox_fit":
            print("[INFO] --input-coord bbox_fit is interpreted as normalized output coordinates.")
        summarize("Assume input is normalized", xyz_as_normalized, low, high)
        n_min, n_mean, n_max = camera_consistency_stats(xyz_as_normalized, camera_models)
        print(
            "\nCamera consistency ratio stats across cameras (uv in [-1,1] and altitude in [min_alt,max_alt]):\n"
            f"- normalized: min={n_min:.4f}, mean={n_mean:.4f}, max={n_max:.4f}"
        )
        return

    if args.input_coord == "utm":
        summarize("Assume input is UTM", xyz_as_utm, low, high)
        u_min, u_mean, u_max = camera_consistency_stats(xyz_as_utm, camera_models)
        print(
            "\nCamera consistency ratio stats across cameras (uv in [-1,1] and altitude in [min_alt,max_alt]):\n"
            f"- utm: min={u_min:.4f}, mean={u_mean:.4f}, max={u_max:.4f}"
        )
        return
    


    summarize("Assume input is normalized", xyz_as_normalized, low, high)
    summarize("Assume input is UTM", xyz_as_utm, low, high)
    n_min, n_mean, n_max = camera_consistency_stats(xyz_as_normalized, camera_models)
    u_min, u_mean, u_max = camera_consistency_stats(xyz_as_utm, camera_models)
    print(
        "\nCamera consistency ratio stats across cameras (uv in [-1,1] and altitude in [min_alt,max_alt]):\n"
        f"- normalized: min={n_min:.4f}, mean={n_mean:.4f}, max={n_max:.4f}\n"
        f"- utm:        min={u_min:.4f}, mean={u_mean:.4f}, max={u_max:.4f}"
    )

    r_norm = ratio_in_bounds(xyz_as_normalized, low, high)
    r_utm = ratio_in_bounds(xyz_as_utm, low, high)
    if max(r_norm, r_utm) < 1e-3:
        print(
            "\nSuggestion: neither mode aligns with affine bounds.\n"
            "- Check that --affine-models-json points to data/affine_models/<SCENE>/affine_models.json\n"
            "- Your cloud may be in another frame (e.g. ENU), which needs an extra transform before conversion.\n"
            "- You can try converter fallback: --input-coord bbox_fit"
        )
    elif r_utm > r_norm:
        print("\nSuggestion: use --input-coord utm")
    elif r_norm > r_utm:
        print("\nSuggestion: use --input-coord normalized")
    else:
        print("\nSuggestion: both modes look similar, inspect min/max and visual results.")


if __name__ == "__main__":
    main()
