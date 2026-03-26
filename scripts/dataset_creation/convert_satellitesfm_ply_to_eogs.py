import argparse
import json
import os
from typing import Optional, Tuple
import numpy as np
from plyfile import PlyData, PlyElement
from pyproj import Transformer
import utm


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



    # return center, scale, min_world, max_world
    expected_zone_number = model.get("n", None)
    expected_zone_letter = model.get("l", None)
    camera_models = []
    for m in metadatas:
        if "model" not in m or "coef_" not in m["model"] or "intercept_" not in m["model"]:
            continue
        if "min_alt" not in m or "max_alt" not in m:
            continue
        A = np.asarray(m["model"]["coef_"], dtype=np.float64)
        b = np.asarray(m["model"]["intercept_"], dtype=np.float64)
        altitude_bounds = np.asarray([m["min_alt"], m["max_alt"]], dtype=np.float64)
        camera_models.append((A, b, altitude_bounds))
    return center, scale, min_world, max_world, camera_models, expected_zone_number, expected_zone_letter



def read_ply_xyz_rgb(path: str):
    ply = PlyData.read(path)
    vertices = ply["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)

    if {"red", "green", "blue"}.issubset(set(vertices.data.dtype.names)):
        rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float64)
    else:
        rgb = np.full((xyz.shape[0], 3), 255.0, dtype=np.float64)
    return xyz, rgb

def load_enu_observer(enu_observer_json: str) -> Tuple[float, float, float]:
    with open(enu_observer_json, "r") as f:
        observer = json.load(f)

    if isinstance(observer, (list, tuple)) and len(observer) >= 3:
        lat, lon, alt = observer[:3]
        return float(lat), float(lon), float(alt)

    if not isinstance(observer, dict):
        raise RuntimeError(
            f"Unsupported ENU observer format in {enu_observer_json}; expected dict or [lat, lon, alt]"
        )

    def pick(keys):
        for k in keys:
            if k in observer:
                return observer[k]
        return None

    lat = pick(["lat", "latitude", "observer_lat", "observer_latitude"])
    lon = pick(["lon", "lng", "longitude", "observer_lon", "observer_longitude"])
    alt = pick(["alt", "altitude", "height", "observer_alt", "observer_altitude"])
    if lat is None or lon is None or alt is None:
        raise RuntimeError(
            f"Cannot parse ENU observer from {enu_observer_json}. "
            "Expected keys like lat/lon/alt (or latitude/longitude/altitude)."
        )
    return float(lat), float(lon), float(alt)


def enu_to_normalized_utm(
    xyz_enu: np.ndarray,
    observer_lat_deg: float,
    observer_lon_deg: float,
    observer_alt_m: float,
    center_utm: np.ndarray,
    scale: float,
) -> np.ndarray:
    # ENU -> ECEF (observer-centered)
    lat = np.deg2rad(observer_lat_deg)
    lon = np.deg2rad(observer_lon_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)

    # Columns correspond to local axes [E, N, U] in ECEF basis
    r_ecef_enu = np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat],
        ],
        dtype=np.float64,
    )

    geodetic_to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
    ecef_to_geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
    x0, y0, z0 = geodetic_to_ecef.transform(observer_lon_deg, observer_lat_deg, observer_alt_m)
    ecef = xyz_enu @ r_ecef_enu.T + np.array([x0, y0, z0], dtype=np.float64)[None]

    lon_deg, lat_deg, alt = ecef_to_geodetic.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
    x_utm, y_utm, _, _ = utm.from_latlon(lat_deg, lon_deg)
    xyz_utm = np.stack([x_utm, y_utm, alt], axis=1).astype(np.float64)
    return (xyz_utm - center_utm[None]) / scale


def infer_utm_zones(lat_deg: np.ndarray, lon_deg: np.ndarray, max_samples: int = 4096):
    n = lat_deg.shape[0]
    if n == 0:
        return set()
    if n > max_samples:
        idx = np.linspace(0, n - 1, num=max_samples, dtype=np.int64)
        lat_s = lat_deg[idx]
        lon_s = lon_deg[idx]
    else:
        lat_s = lat_deg
        lon_s = lon_deg
    zones = set()
    for la, lo in zip(lat_s.tolist(), lon_s.tolist()):
        _, _, zn, zl = utm.from_latlon(float(la), float(lo))
        zones.add((int(zn), str(zl)))
    return zones

def write_ply_xyz_rgb(path: str, xyz: np.ndarray, rgb: np.ndarray):




    # os.makedirs(os.path.dirname(path), exist_ok=True)
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)






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
        # choices=["utm", "normalized", "auto", "bbox_fit"],
         choices=["utm", "normalized", "auto", "bbox_fit", "enu"],
        default="utm",
        help=(
            "Coordinate system of input points. "
            "'utm' applies EOGS normalization (x-center)/scale. "
            "'enu' converts from a local ENU frame using --enu-observer-json and then normalizes to EOGS world. "
            "'auto' picks the mode that better matches affine scene bounds. "
            "'bbox_fit' ignores absolute georeferencing and maps input cloud bbox to affine scene bbox."
        ),
    )
    parser.add_argument(
        "--enu-observer-json",
        type=str,
        default=None,
        help=(
            "Path to ENU observer geodetic origin JSON (lat/lon/alt), e.g. SatelliteSfM "
            "enu_observer_latlonalt.json. Required for --input-coord enu; used by --input-coord auto when provided."
        ),
    )
    parser.add_argument(
        "--allow-enu-unsafe",
        action="store_true",
        help=(
            "Allow ENU conversion even when heuristics suggest risky setup "
            "(large area / weak camera consistency). Use with caution."
        ),
    )
    parser.add_argument(
        "--enu-max-horizontal-extent-m",
        type=float,
        default=1000.0,
        help="Safety threshold: max recommended ENU horizontal extent in meters.",
    )
    parser.add_argument(
        "--enu-min-camera-mean",
        type=float,
        default=0.20,
        help="Safety threshold: minimum camera-consistency mean expected after ENU conversion.",
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
        "--bbox-fit-isotropic-axes",
        choices=["xy", "xyz"],
        default="xy",
        help=(
            "For isotropic bbox fit, which axes are used to estimate the single scale. "
            "'xy' (default) avoids Z-range domination for satellite clouds; 'xyz' uses all axes."
        ),
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




    parser.add_argument(
        "--dry-run-report",
        action="store_true",
        help="Print detailed diagnostics for coordinate-mode scoring before writing output.",
    )
    parser.add_argument(
        "--disable-camera-safety-filter",
        action="store_true",
        help="Disable point filtering based on affine camera UV/altitude consistency (not recommended).",
    )
    parser.add_argument(
        "--camera-safety-uv-margin",
        type=float,
        default=1.2,
        help="UV bound for camera safety filter (keeps points with |u|,|v| <= margin in at least one camera).",
    )
    parser.add_argument(
        "--camera-safety-alt-margin",
        type=float,
        default=5.0,
        help="Altitude margin added to [min_alt,max_alt] in camera safety filter.",
    )
    parser.add_argument(
        "--max-camera-altitude",
        type=float,
        default=180.0,
        help=(
            "Hard ceiling on camera-space altitude used by safety filter. "
            "Helps avoid rasterizer crashes from extremely high points."
        ),
    )




    args = parser.parse_args()

    # center, scale, min_world, max_world = load_affine_normalization(args.affine_models_json)
    # center, scale, min_world, max_world, camera_models = load_affine_normalization(args.affine_models_json)
    loaded = load_affine_normalization(args.affine_models_json)
    # Backward-compatible unpacking in case local branches still use the older
    # 5-value return signature.
    if len(loaded) == 5:
        center, scale, min_world, max_world, camera_models = loaded
        expected_zone_number, expected_zone_letter = None, None
    else:
        center, scale, min_world, max_world, camera_models, expected_zone_number, expected_zone_letter = loaded

    
    
    
    xyz, rgb = read_ply_xyz_rgb(args.input_ply)

    def in_bounds_ratio(pts: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        return float(np.mean(np.all((pts >= lower[None]) & (pts <= upper[None]), axis=1)))
    




    def camera_consistency_ratio(pts_world: np.ndarray, A: np.ndarray, b: np.ndarray, altitude_bounds: np.ndarray):
        uva = pts_world @ A.T + b[None]
        uv_ok = np.all(np.abs(uva[:, :2]) <= 1.0, axis=1)
        alt_ok = (uva[:, 2] >= altitude_bounds[0]) & (uva[:, 2] <= altitude_bounds[1])
        return float(np.mean(uv_ok & alt_ok))

    def camera_consistency_stats(pts_world: np.ndarray):
        if len(camera_models) == 0:
            return float("nan"), float("nan"), float("nan")
        ratios = [
            camera_consistency_ratio(pts_world, A, b, altitude_bounds)
            for A, b, altitude_bounds in camera_models
        ]
        ratios = np.asarray(ratios, dtype=np.float64)
        return float(ratios.min()), float(ratios.mean()), float(ratios.max())

    def finite_or_zero(x: float):
        return 0.0 if not np.isfinite(x) else float(x)

    def debug_candidate(name: str, pts_world: np.ndarray, lower: np.ndarray, upper: np.ndarray):
        r = in_bounds_ratio(pts_world, lower, upper)
        cmin, cmean, cmax = camera_consistency_stats(pts_world)
        if len(pts_world) == 0:
            print(f"[AUTO][REPORT] {name}: empty point set")
            return
        print(
            f"[AUTO][REPORT] {name}: in_bounds={r:.6f}, cam(min/mean/max)=({cmin:.6f}/{cmean:.6f}/{cmax:.6f}), "
            f"min={pts_world.min(axis=0)}, max={pts_world.max(axis=0)}"
        )





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
            # fit_scale = args.bbox_fit_fill_ratio * np.min(dst_size / src_size)
            if args.bbox_fit_isotropic_axes == "xy":
                fit_scale = args.bbox_fit_fill_ratio * np.min(dst_size[:2] / src_size[:2])
            else:
                fit_scale = args.bbox_fit_fill_ratio * np.min(dst_size / src_size)
            fitted = (src - src_center[None]) * fit_scale + dst_center[None]
            z_span = float(np.max(fitted[:, 2]) - np.min(fitted[:, 2]))
            dst_z_span = float(dst_size[2])
            print(
                # f"[BBOX_FIT] fit_scale={fit_scale:.6e}, "
                # f"src_size={src_size}, dst_size={dst_size}, fill_ratio={args.bbox_fit_fill_ratio}"
                f"[BBOX_FIT] fit_scale={fit_scale:.6e}, axes={args.bbox_fit_isotropic_axes}, "
                f"src_size={src_size}, dst_size={dst_size}, fill_ratio={args.bbox_fit_fill_ratio}, "
                f"z_span={z_span:.6f}"
            )
            if dst_z_span > 0 and z_span > 2.0 * dst_z_span:
                print(
                    "[BBOX_FIT][WARN] Converted Z span is much larger than affine bbox Z span. "
                    "If rendering looks unstable, try --bbox-fit-anisotropic or reduce fill ratio."
                )
        return fitted
    
    xyz_world_enu: Optional[np.ndarray] = None
    if args.enu_observer_json is not None:
        enu_extent_xy = np.max(xyz[:, :2], axis=0) - np.min(xyz[:, :2], axis=0)
        max_extent_xy = float(np.max(enu_extent_xy))
        if max_extent_xy > args.enu_max_horizontal_extent_m and not args.allow_enu_unsafe:
            raise RuntimeError(
                "ENU conversion safety check failed: horizontal extent is too large "
                f"({max_extent_xy:.2f}m > {args.enu_max_horizontal_extent_m:.2f}m).\n"
                "For large-area scenes, direct geodetic integration is recommended instead of ENU conversion chain.\n"
                "If you still want to proceed, pass --allow-enu-unsafe."
            )
        obs_lat, obs_lon, obs_alt = load_enu_observer(args.enu_observer_json)
        xyz_world_enu = enu_to_normalized_utm(
            xyz_enu=xyz,
            observer_lat_deg=obs_lat,
            observer_lon_deg=obs_lon,
            observer_alt_m=obs_alt,
            center_utm=center,
            scale=scale,
        )
        # UTM zone consistency check (against affine model zone metadata, if available)
        geodetic_to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
        ecef_to_geodetic = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
        lat = np.deg2rad(obs_lat)
        lon = np.deg2rad(obs_lon)
        sin_lat, cos_lat = np.sin(lat), np.cos(lat)
        sin_lon, cos_lon = np.sin(lon), np.cos(lon)
        r_ecef_enu = np.array(
            [
                [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
                [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
                [0.0, cos_lat, sin_lat],
            ],
            dtype=np.float64,
        )
        x0, y0, z0 = geodetic_to_ecef.transform(obs_lon, obs_lat, obs_alt)
        ecef = xyz @ r_ecef_enu.T + np.array([x0, y0, z0], dtype=np.float64)[None]
        lon_deg, lat_deg, _ = ecef_to_geodetic.transform(ecef[:, 0], ecef[:, 1], ecef[:, 2])
        zones = infer_utm_zones(np.asarray(lat_deg), np.asarray(lon_deg))
        if len(zones) > 1 and not args.allow_enu_unsafe:
            raise RuntimeError(
                f"ENU conversion safety check failed: points span multiple UTM zones {sorted(zones)}.\n"
                "This can cause severe frame mismatch. Use direct geodetic integration or split the scene.\n"
                "If you still want to proceed, pass --allow-enu-unsafe."
            )
        if expected_zone_number is not None and expected_zone_letter is not None and len(zones) > 0:
            expected_zone = (int(expected_zone_number), str(expected_zone_letter))
            if expected_zone not in zones and not args.allow_enu_unsafe:
                raise RuntimeError(
                    f"ENU conversion safety check failed: inferred zone(s) {sorted(zones)} do not include "
                    f"affine model zone {expected_zone}.\n"
                    "This usually indicates wrong observer origin or scene mismatch.\n"
                    "If you still want to proceed, pass --allow-enu-unsafe."
                )

    if args.input_coord == "utm":
        xyz_world = (xyz - center[None]) / scale
    elif args.input_coord == "normalized":
        xyz_world = xyz.copy()
    elif args.input_coord == "enu":
        if xyz_world_enu is None:
            raise RuntimeError("--input-coord enu requires --enu-observer-json")
        ratio_enu_only = in_bounds_ratio(xyz_world_enu, min_world - 0.10 * (max_world - min_world), max_world + 0.10 * (max_world - min_world))
        _, cam_mean_enu_only, _ = camera_consistency_stats(xyz_world_enu)
        if cam_mean_enu_only < args.enu_min_camera_mean and not args.allow_enu_unsafe:
            raise RuntimeError(
                "ENU conversion safety check failed: low camera consistency after conversion "
                f"(mean={cam_mean_enu_only:.4f} < {args.enu_min_camera_mean:.4f}, "
                f"in-bounds={ratio_enu_only:.4f}).\n"
                "This often indicates frame/baseline mismatch across conversion chain. "
                "Recalibrate conversion or use direct geodetic integration.\n"
                "If you still want to proceed, pass --allow-enu-unsafe."
            )
        xyz_world = xyz_world_enu
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




        _, cam_mean_utm, _ = camera_consistency_stats(xyz_world_utm)
        _, cam_mean_norm, _ = camera_consistency_stats(xyz_world_norm)
        ratio_enu = -1.0
        cam_mean_enu = float("nan")
        score_enu = -1.0

        # Weighted score: bbox overlap is robust and cheap; camera consistency
        # disambiguates cases where bbox overlap is similar but rendering fails.
        score_utm = 0.6 * ratio_utm + 0.4 * finite_or_zero(cam_mean_utm)
        score_norm = 0.6 * ratio_norm + 0.4 * finite_or_zero(cam_mean_norm)
        if xyz_world_enu is not None:
            ratio_enu = in_bounds_ratio(xyz_world_enu, lower_auto, upper_auto)
            _, cam_mean_enu, _ = camera_consistency_stats(xyz_world_enu)
            score_enu = 0.6 * ratio_enu + 0.4 * finite_or_zero(cam_mean_enu)

        if args.dry_run_report:
            msg = (
                "[AUTO][REPORT] "
                f"utm: ratio={ratio_utm:.6f}, cam_mean={cam_mean_utm:.6f}, score={score_utm:.6f}; "
                f"normalized: ratio={ratio_norm:.6f}, cam_mean={cam_mean_norm:.6f}, score={score_norm:.6f}"
            )
            if xyz_world_enu is not None:
                msg += f"; enu: ratio={ratio_enu:.6f}, cam_mean={cam_mean_enu:.6f}, score={score_enu:.6f}"
            print(msg)
            debug_candidate("utm", xyz_world_utm, lower_auto, upper_auto)
            debug_candidate("normalized", xyz_world_norm, lower_auto, upper_auto)
            if xyz_world_enu is not None:
                debug_candidate("enu", xyz_world_enu, lower_auto, upper_auto)

        if max(ratio_utm, ratio_norm, ratio_enu) < 1e-3:





            if args.auto_fallback == "bbox_fit":
                print(
                    # "[AUTO] neither 'utm' nor 'normalized' matched scene bounds "
                    # f"(utm={ratio_utm:.6f}, normalized={ratio_norm:.6f}); "
                    "[AUTO] none of candidate coordinate modes matched scene bounds "
                    f"(utm={ratio_utm:.6f}, normalized={ratio_norm:.6f}, enu={ratio_enu:.6f}); "
                    "falling back to bbox_fit."
                )
                xyz_world = apply_bbox_fit(xyz)
            else:
                raise RuntimeError(
                    # "AUTO mode failed: both 'utm' and 'normalized' have near-zero in-bounds ratio.\n"
                    "AUTO mode failed: all candidate modes have near-zero in-bounds ratio.\n"
                    "Please check that --affine-models-json points to data/affine_models/<SCENE>/affine_models.json\n"
                    "and verify whether your input cloud is in another frame (e.g. ENU) requiring extra conversion.\n"
                    # "Tip: rerun with --auto-fallback bbox_fit or --input-coord bbox_fit."
                    "Tip: if cloud is ENU provide --enu-observer-json; otherwise rerun with --auto-fallback bbox_fit."
                )
        # if ratio_utm >= ratio_norm:
        # if ratio_utm >= score_norm:
        #     if max(ratio_utm, ratio_norm) >= 1e-3:
        elif score_enu >= score_utm and score_enu >= score_norm and xyz_world_enu is not None:
            if cam_mean_enu < args.enu_min_camera_mean and not args.allow_enu_unsafe:
                raise RuntimeError(
                    "AUTO selected ENU candidate but failed ENU safety check: "
                    f"camera consistency mean={cam_mean_enu:.4f} < threshold={args.enu_min_camera_mean:.4f}.\n"
                    "Use direct geodetic integration or pass --allow-enu-unsafe to override."
                )
            xyz_world = xyz_world_enu
            print(
                "[AUTO] selected input-coord=enu "
                f"(score={score_enu:.4f}, utm_score={score_utm:.4f}, normalized_score={score_norm:.4f}, "
                f"in-bounds ratio={ratio_enu:.4f}, cam-mean={cam_mean_enu:.4f})"
            )
        elif score_utm >= score_norm:
            if max(ratio_utm, ratio_norm, ratio_enu) >= 1e-3:
                xyz_world = xyz_world_utm
                # print(f"[AUTO] selected input-coord=utm (in-bounds ratio={ratio_utm:.4f}, normalized={ratio_norm:.4f})")
                print(
                    "[AUTO] selected input-coord=utm "
                    f"(score={score_utm:.4f}, normalized_score={score_norm:.4f}, "
                    f"in-bounds ratio={ratio_utm:.4f}, cam-mean={cam_mean_utm:.4f})"
                )
        else:
            if max(ratio_utm, ratio_norm, ratio_enu) >= 1e-3:
                xyz_world = xyz_world_norm
                # print(f"[AUTO] selected input-coord=normalized (in-bounds ratio={ratio_norm:.4f}, utm={ratio_utm:.4f})")
                print(
                    "[AUTO] selected input-coord=normalized "
                    f"(score={score_norm:.4f}, utm_score={score_utm:.4f}, "
                    f"in-bounds ratio={ratio_norm:.4f}, cam-mean={cam_mean_norm:.4f})"
                )
    if not args.disable_camera_safety_filter and len(camera_models) > 0 and len(xyz_world) > 0:
        keep = np.zeros((xyz_world.shape[0],), dtype=bool)
        for A, b, altitude_bounds in camera_models:
            uva = xyz_world @ A.T + b[None]
            uv_ok = np.all(np.abs(uva[:, :2]) <= args.camera_safety_uv_margin, axis=1)
            alt_low = altitude_bounds[0] - args.camera_safety_alt_margin
            alt_high = min(altitude_bounds[1] + args.camera_safety_alt_margin, args.max_camera_altitude)
            alt_ok = (uva[:, 2] >= alt_low) & (uva[:, 2] <= alt_high)
            keep |= (uv_ok & alt_ok)
        kept = int(np.sum(keep))
        removed = int(len(keep) - kept)
        print(
            "[SAFETY_FILTER] camera UV/altitude filtering "
            f"kept={kept}, removed={removed}, "
            f"uv_margin={args.camera_safety_uv_margin}, alt_margin={args.camera_safety_alt_margin}, "
            f"max_camera_altitude={args.max_camera_altitude}"
        )
        xyz_world = xyz_world[keep]
        rgb = rgb[keep]

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
