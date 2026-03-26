import argparse
import json
import os

import numpy as np
from plyfile import PlyData, PlyElement


def load_affine_normalization(affine_models_json: str):
    with open(affine_models_json, "r") as f:
        metadatas = json.load(f)
    if len(metadatas) == 0:
        raise RuntimeError(f"No camera metadata found in {affine_models_json}")

    # Same normalization for all cameras in a scene, use first entry.
    model = metadatas[0]["model"]
    center = np.asarray(model["center"], dtype=np.float64)
    scale = float(model["scale"])
    min_world = np.asarray(model["min_world"], dtype=np.float64)
    max_world = np.asarray(model["max_world"], dtype=np.float64)
    return center, scale, min_world, max_world


def read_ply_xyz_rgb(path: str):
    ply = PlyData.read(path)
    vertices = ply["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)

    if {"red", "green", "blue"}.issubset(set(vertices.data.dtype.names)):
        rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float64)
    else:
        rgb = np.full((xyz.shape[0], 3), 255.0, dtype=np.float64)
    return xyz, rgb


def write_ply_xyz_rgb(path: str, xyz: np.ndarray, rgb: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "f4"),
        ("green", "f4"),
        ("blue", "f4"),
    ]
    normals = np.zeros_like(xyz, dtype=np.float32)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate([xyz.astype(np.float32), normals, rgb.astype(np.float32)], axis=1)
    elements[:] = list(map(tuple, attributes))
    ply = PlyData([PlyElement.describe(elements, "vertex")])
    ply.write(path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SatelliteSfM point cloud to EOGS normalized affine world coordinates."
    )
    parser.add_argument("--input-ply", required=True, help="Input sparse point cloud .ply from SatelliteSfM")
    parser.add_argument(
        "--affine-models-json",
        required=True,
        help="Path to affine_models.json (e.g. data/affine_models/JAX_004/affine_models.json)",
    )
    parser.add_argument(
        "--output-ply",
        required=True,
        help="Output .ply path (usually data/affine_models/<scene>/points3d.ply)",
    )
    parser.add_argument(
        "--input-coord",
        choices=["utm", "normalized", "auto", "bbox_fit"],
        default="utm",
        help=(
            "Coordinate system of input points. "
            "'utm' applies EOGS normalization (x-center)/scale. "
            "'auto' picks the mode that better matches affine scene bounds. "
            "'bbox_fit' ignores absolute georeferencing and maps input cloud bbox to affine scene bbox."
        ),
    )
    parser.add_argument(
        "--auto-fallback",
        choices=["error", "bbox_fit"],
        default="bbox_fit",
        help=(
            "Behavior when --input-coord auto cannot match either 'utm' or 'normalized'. "
            "'bbox_fit' falls back to bbox fitting automatically; 'error' raises an exception."
        ),
    )
    parser.add_argument(
        "--bbox-fit-fill-ratio",
        type=float,
        default=0.95,
        help="For --input-coord bbox_fit, target fraction of affine bbox occupied by the fitted cloud.",
    )
    parser.add_argument(
        "--bbox-fit-pca-align",
        action="store_true",
        help="For --input-coord bbox_fit, rotate by PCA before fitting bbox (useful if source cloud is arbitrarily rotated).",
    )
    parser.add_argument(
        "--bbox-fit-anisotropic",
        action="store_true",
        help="For --input-coord bbox_fit, use per-axis scaling instead of isotropic scaling.",
    )
    parser.add_argument(
        "--crop-to-scene-bounds",
        action="store_true",
        help="Crop points to [min_world,max_world] from affine_models.json (with margin).",
    )
    parser.add_argument(
        "--bounds-margin",
        type=float,
        default=0.10,
        help="Relative margin when --crop-to-scene-bounds is enabled.",
    )
    parser.add_argument(
        "--allow-empty-output",
        action="store_true",
        help="Allow writing an empty output point cloud (normally treated as an error).",
    )
    args = parser.parse_args()

    center, scale, min_world, max_world = load_affine_normalization(args.affine_models_json)
    xyz, rgb = read_ply_xyz_rgb(args.input_ply)

    def in_bounds_ratio(pts: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        return float(np.mean(np.all((pts >= lower[None]) & (pts <= upper[None]), axis=1)))

    def apply_bbox_fit(src_xyz: np.ndarray):
        src = src_xyz.copy()
        if args.bbox_fit_pca_align:
            src_center = src.mean(axis=0)
            src_centered = src - src_center[None]
            cov = np.cov(src_centered, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            R = evecs[:, order]
            src = src_centered @ R
        src_min = src.min(axis=0)
        src_max = src.max(axis=0)
        src_center = 0.5 * (src_min + src_max)
        src_size = np.maximum(src_max - src_min, 1e-12)

        dst_min = min_world
        dst_max = max_world
        dst_center = 0.5 * (dst_min + dst_max)
        dst_size = np.maximum(dst_max - dst_min, 1e-12)
        if args.bbox_fit_anisotropic:
            fit_scale = args.bbox_fit_fill_ratio * (dst_size / src_size)
            fitted = (src - src_center[None]) * fit_scale[None] + dst_center[None]
            print(
                f"[BBOX_FIT][ANISO] fit_scale={fit_scale}, "
                f"src_size={src_size}, dst_size={dst_size}, fill_ratio={args.bbox_fit_fill_ratio}"
            )
        else:
            fit_scale = args.bbox_fit_fill_ratio * np.min(dst_size / src_size)
            fitted = (src - src_center[None]) * fit_scale + dst_center[None]
            print(
                f"[BBOX_FIT] fit_scale={fit_scale:.6e}, "
                f"src_size={src_size}, dst_size={dst_size}, fill_ratio={args.bbox_fit_fill_ratio}"
            )
        return fitted

    if args.input_coord == "utm":
        xyz_world = (xyz - center[None]) / scale
    elif args.input_coord == "normalized":
        xyz_world = xyz.copy()
    elif args.input_coord == "bbox_fit":
        xyz_world = apply_bbox_fit(xyz)
    else:
        extent = max_world - min_world
        lower_auto = min_world - 0.10 * extent
        upper_auto = max_world + 0.10 * extent
        xyz_world_utm = (xyz - center[None]) / scale
        xyz_world_norm = xyz.copy()
        ratio_utm = in_bounds_ratio(xyz_world_utm, lower_auto, upper_auto)
        ratio_norm = in_bounds_ratio(xyz_world_norm, lower_auto, upper_auto)
        if max(ratio_utm, ratio_norm) < 1e-3:
            if args.auto_fallback == "bbox_fit":
                print(
                    "[AUTO] neither 'utm' nor 'normalized' matched scene bounds "
                    f"(utm={ratio_utm:.6f}, normalized={ratio_norm:.6f}); "
                    "falling back to bbox_fit."
                )
                xyz_world = apply_bbox_fit(xyz)
            else:
                raise RuntimeError(
                    "AUTO mode failed: both 'utm' and 'normalized' have near-zero in-bounds ratio.\n"
                    "Please check that --affine-models-json points to data/affine_models/<SCENE>/affine_models.json\n"
                    "and verify whether your input cloud is in another frame (e.g. ENU) requiring extra conversion.\n"
                    "Tip: rerun with --auto-fallback bbox_fit or --input-coord bbox_fit."
                )
        if ratio_utm >= ratio_norm:
            if max(ratio_utm, ratio_norm) >= 1e-3:
                xyz_world = xyz_world_utm
                print(f"[AUTO] selected input-coord=utm (in-bounds ratio={ratio_utm:.4f}, normalized={ratio_norm:.4f})")
        else:
            if max(ratio_utm, ratio_norm) >= 1e-3:
                xyz_world = xyz_world_norm
                print(f"[AUTO] selected input-coord=normalized (in-bounds ratio={ratio_norm:.4f}, utm={ratio_utm:.4f})")

    if args.crop_to_scene_bounds:
        extent = max_world - min_world
        lower = min_world - args.bounds_margin * extent
        upper = max_world + args.bounds_margin * extent
        mask = np.all((xyz_world >= lower[None]) & (xyz_world <= upper[None]), axis=1)
        xyz_world = xyz_world[mask]
        rgb = rgb[mask]

    print(f"Input points: {len(xyz)}")
    print(f"Output points: {len(xyz_world)}")

    if len(xyz_world) == 0:
        msg = (
            "No points left after conversion/cropping.\n"
            "Likely causes:\n"
            "  1) Input coordinate system mismatch (e.g. ENU points but --input-coord utm).\n"
            "  2) Scene bounds too strict for the incoming point cloud.\n"
            "Try one of:\n"
            "  - remove --crop-to-scene-bounds for a quick sanity check\n"
            "  - use --input-coord normalized if your input is already normalized\n"
            "  - increase --bounds-margin\n"
        )
        if not args.allow_empty_output:
            raise RuntimeError(msg)
        print("[WARN]", msg)
    else:
        print(f"Output range min: {xyz_world.min(axis=0)}")
        print(f"Output range max: {xyz_world.max(axis=0)}")

    write_ply_xyz_rgb(args.output_ply, xyz_world, rgb)
    print(f"Saved: {args.output_ply}")


if __name__ == "__main__":
    main()
