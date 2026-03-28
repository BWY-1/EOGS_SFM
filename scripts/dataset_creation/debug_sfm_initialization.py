import argparse
import json
import numpy as np
from plyfile import PlyData


def read_ply_xyz(path: str) -> np.ndarray:
    ply = PlyData.read(path)
    v = ply["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)


def read_affine_models(path: str):
    with open(path, "r") as f:
        metas = json.load(f)
    if len(metas) == 0:
        raise RuntimeError(f"No metadata found in {path}")

    model0 = metas[0]["model"]
    center = np.asarray(model0["center"], dtype=np.float64)
    scale = float(model0["scale"])
    min_world = np.asarray(model0["min_world"], dtype=np.float64)
    max_world = np.asarray(model0["max_world"], dtype=np.float64)

    camera_models = []
    for m in metas:
        if "model" not in m:
            continue
        model = m["model"]
        if "coef_" not in model or "intercept_" not in model:
            continue
        if "min_alt" not in m or "max_alt" not in m:
            continue
        A = np.asarray(model["coef_"], dtype=np.float64)
        b = np.asarray(model["intercept_"], dtype=np.float64)
        alt = np.asarray([m["min_alt"], m["max_alt"]], dtype=np.float64)
        camera_models.append((A, b, alt))

    return center, scale, min_world, max_world, camera_models


def ratio_in_bounds(pts: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean(np.all((pts >= low[None]) & (pts <= high[None]), axis=1)))


def camera_consistency_stats(pts_world: np.ndarray, camera_models):
    if len(camera_models) == 0:
        return float("nan"), float("nan"), float("nan")

    ratios = []
    for A, b, altitude_bounds in camera_models:
        uva = pts_world @ A.T + b[None]
        uv_ok = np.all(np.abs(uva[:, :2]) <= 1.0, axis=1)
        alt_ok = (uva[:, 2] >= altitude_bounds[0]) & (uva[:, 2] <= altitude_bounds[1])
        ratios.append(float(np.mean(uv_ok & alt_ok)))

    ratios = np.asarray(ratios, dtype=np.float64)
    return float(np.min(ratios)), float(np.mean(ratios)), float(np.max(ratios))


def approx_knn_dist2(pts: np.ndarray, sample_size: int = 3000) -> float:
    n = len(pts)
    if n < 2:
        return float("nan")
    if n > sample_size:
        idx = np.random.default_rng(0).choice(n, sample_size, replace=False)
        pts = pts[idx]

    d2_min = np.full((len(pts),), np.inf, dtype=np.float64)
    block = 1024
    for i in range(0, len(pts), block):
        a = pts[i:i + block]
        d2 = np.sum((a[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        row_ids = np.arange(len(a))
        col_ids = np.arange(i, min(i + block, len(pts)))
        d2[row_ids, col_ids] = np.inf
        d2_min[i:i + block] = np.min(d2, axis=1)

    return float(np.median(d2_min))


def print_mode_report(name: str, pts_world: np.ndarray, low: np.ndarray, high: np.ndarray, camera_models):
    inb = ratio_in_bounds(pts_world, low, high)
    cmin, cmean, cmax = camera_consistency_stats(pts_world, camera_models)
    print(f"[{name}]")
    print(f"  in_bounds_ratio: {inb:.6f}")
    print(f"  camera_consistency(min/mean/max): {cmin:.6f} / {cmean:.6f} / {cmax:.6f}")
    print(f"  min_xyz: {pts_world.min(axis=0)}")
    print(f"  max_xyz: {pts_world.max(axis=0)}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose point-cloud initialization quality for EOGS affine training.")
    parser.add_argument("--input-ply", required=True)
    parser.add_argument("--affine-models-json", required=True)
    parser.add_argument("--bounds-margin", type=float, default=0.10)
    args = parser.parse_args()

    xyz = read_ply_xyz(args.input_ply)
    center, scale, min_world, max_world, camera_models = read_affine_models(args.affine_models_json)

    if len(xyz) == 0:
        raise RuntimeError("Input PLY has 0 points")

    extent = max_world - min_world
    low = min_world - args.bounds_margin * extent
    high = max_world + args.bounds_margin * extent

    xyz_norm = xyz
    xyz_utm = (xyz - center[None]) / scale

    print("=== Coordinate-mode diagnostics ===")
    print_mode_report("assume_normalized", xyz_norm, low, high, camera_models)
    print_mode_report("assume_utm", xyz_utm, low, high, camera_models)

    print("\n=== Density diagnostics (current PLY coordinates) ===")
    med_d2 = approx_knn_dist2(xyz)
    if np.isfinite(med_d2):
        med_dist = float(np.sqrt(med_d2))
        print(f"  median_nn_dist: {med_dist:.6e}")
        print(f"  suggested_init_scale_multiplier (rough): 0.5 ~ 2.0 around {med_dist:.6e}")
    else:
        print("  median_nn_dist: NaN (not enough points)")

    print("\n=== Quick interpretation ===")
    norm_inb = ratio_in_bounds(xyz_norm, low, high)
    utm_inb = ratio_in_bounds(xyz_utm, low, high)
    if max(norm_inb, utm_inb) < 1e-3:
        print("  - Neither normalized nor UTM aligns with affine bounds. Try converter --input-coord enu/auto/bbox_fit.")
    elif utm_inb > norm_inb:
        print("  - UTM interpretation looks better. Prefer converter mode --input-coord utm (or auto).")
    elif norm_inb > utm_inb:
        print("  - Normalized interpretation looks better. Prefer --input-coord normalized.")
    else:
        print("  - Both modes are close; compare camera consistency means to choose.")


if __name__ == "__main__":
    main()
