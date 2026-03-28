# Sparse Initialization A/B Matrix (EOGS + SatelliteSfM)

This checklist isolates whether blur comes from sparse initialization density rather than coordinate mismatch.

## Preconditions
- Converted point cloud already validated in affine-normalized frame (e.g. `check_affine_pointcloud_alignment.py` reports normalized in-bounds near 1.0).
- Use the exact same dataset split and training images for all runs.

## Common variables
```bash
SCENE=JAX_068
SRC=/home/m/EOGS-SFM/data/affine_models/${SCENE}
IMGS=/home/m/EOGS-SFM/data/${SCENE}/images
OUT=/home/m/EOGS-SFM/output
```

## Run A (baseline sparse only)
```bash
cd src/gaussiansplatting
python train.py \
  -s ${SRC} \
  --images ${IMGS} \
  --eval \
  -m ${OUT}/${SCENE,,}_ablate_A_sparse \
  --sh_degree 0 \
  --iterations 7000
```

## Run B (recommended denser seed)
```bash
cd src/gaussiansplatting
python train.py \
  -s ${SRC} \
  --images ${IMGS} \
  --eval \
  -m ${OUT}/${SCENE,,}_ablate_B_seed120k \
  --sh_degree 0 \
  --iterations 7000 \
  --init-random-points 120000 \
  --init-random-points-bbox-scale 1.05 \
  --init-scale-ceiling 0.03
```

## Run C (higher seed + looser ceiling)
```bash
cd src/gaussiansplatting
python train.py \
  -s ${SRC} \
  --images ${IMGS} \
  --eval \
  -m ${OUT}/${SCENE,,}_ablate_C_seed200k \
  --sh_degree 0 \
  --iterations 7000 \
  --init-random-points 200000 \
  --init-random-points-bbox-scale 1.05 \
  --init-scale-ceiling 0.05
```

## Fast compare at key iterations
Compare qualitative outputs at 1000 / 3000 / 7000 iterations and record:
- geometry sharpness (building edges)
- floaters/background haze
- validation PSNR/SSIM if available

Tip: if B improves a lot over A, sparse density was the main bottleneck.
