#!/usr/bin/env python3
import argparse
import numpy as np
import iio


def to_uint8(img: np.ndarray, gamma: float = 1.0, percentile: float | None = None) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
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
        "--percentile",
        type=float,
        default=None,
        help="Optional symmetric percentile stretch (e.g. 1.0 means [1,99] percentiles).",
    )
    args = parser.parse_args()

    img = iio.read(args.input)
    arr = np.asarray(img)
    print(f"Loaded: {args.input}")
    print(f"shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")

    png = to_uint8(arr, gamma=args.gamma, percentile=args.percentile)
    iio.write(args.output, png)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
