#!/usr/bin/env python3
import argparse
import numpy as np
import iio


def stats(name, x):
    x = np.asarray(x, dtype=np.float32)
    print(f"[{name}] shape={x.shape}, dtype={x.dtype}")
    for c, n in enumerate(["R", "G", "B"]):
        ch = x[..., c]
        p = np.percentile(ch, [1, 5, 50, 95, 99])
        print(
            f"  {n}: min={ch.min():.4f} mean={ch.mean():.4f} max={ch.max():.4f} "
            f"p1/p5/p50/p95/p99={p[0]:.4f}/{p[1]:.4f}/{p[2]:.4f}/{p[3]:.4f}/{p[4]:.4f}"
        )


def linear_fit(src, tgt):
    # Solve tgt ≈ a * src + b channel-wise.
    a = np.zeros(3, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    for c in range(3):
        x = src[..., c].reshape(-1).astype(np.float64)
        y = tgt[..., c].reshape(-1).astype(np.float64)
        A = np.stack([x, np.ones_like(x)], axis=1)
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a[c] = float(coef[0])
        b[c] = float(coef[1])
    return a, b


def main():
    p = argparse.ArgumentParser(description="Compare tone statistics between render and GT .iio files.")
    p.add_argument("--render", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--save-corrected", default=None, help="Optional output .png for linearly corrected render preview")
    args = p.parse_args()

    r = np.asarray(iio.read(args.render), dtype=np.float32)
    g = np.asarray(iio.read(args.gt), dtype=np.float32)
    if r.shape != g.shape or r.ndim != 3 or r.shape[2] != 3:
        raise RuntimeError(f"Expected matching HxWx3 shapes, got render={r.shape}, gt={g.shape}")

    r = np.clip(r, 0.0, 1.0)
    g = np.clip(g, 0.0, 1.0)

    stats("render", r)
    stats("gt", g)

    a, b = linear_fit(r, g)
    print("\nLinear color fit (gt ≈ a*render + b):")
    print(f"  a={a}")
    print(f"  b={b}")

    if args.save_corrected:
        corr = np.clip(r * a[None, None, :] + b[None, None, :], 0.0, 1.0)
        iio.write(args.save_corrected, (corr * 255.0 + 0.5).astype(np.uint8))
        print(f"Saved corrected preview: {args.save_corrected}")


if __name__ == "__main__":
    main()
