#!/usr/bin/env python3
import argparse
from typing import Optional

import numpy as np

from iio_compat import load_array, save_uint8_image


def to_uint8(
    img: np.ndarray,
    gamma: float = 1.0,
    percentile: Optional[float] = None,
    normalize: str = "clip",
) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if normalize == "max":
        white_point = float(x.max())
        x = x / max(white_point, 1e-8)
    elif normalize == "p99":
        white_point = float(np.percentile(x, 99.0))
        x = x / max(white_point, 1e-8)

    if percentile is not None:
        lo = np.percentile(x, percentile)
        hi = np.percentile(x, 100.0 - percentile)
        x = np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    else:
        x = np.clip(x, 0.0, 1.0)

    if gamma != 1.0:
        x = np.power(x, gamma)

    return (x * 255.0 + 0.5).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Convert EOGS .iio float image to display PNG safely.")
    parser.add_argument("--input", required=True, help="Input .iio path")
    parser.add_argument("--output", required=True, help="Output .png path")
    parser.add_argument("--gamma", type=float, default=1.0, help="Display gamma. Keep 1.0 unless needed.")
    parser.add_argument(
        "--normalize",
        choices=["clip", "max", "p99"],
        default="clip",
        help="Display normalization before clipping: keep unit range, divide by global max, or divide by p99.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Optional symmetric percentile stretch (e.g. 1.0 means [1,99] percentiles).",
    )
    args = parser.parse_args()

    img = load_array(args.input)
    arr = np.asarray(img)
    print(f"Loaded: {args.input}")
    print(f"shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")
    print(
        "outside_[0,1]="
        f"{((arr < 0.0) | (arr > 1.0)).mean():.6f}, "
        f"below_0={(arr < 0.0).mean():.6f}, above_1={(arr > 1.0).mean():.6f}"
    )

    png = to_uint8(arr, gamma=args.gamma, percentile=args.percentile, normalize=args.normalize)
    save_uint8_image(args.output, png)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
