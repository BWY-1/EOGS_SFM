# Sparse-init Finetune (based on B seed setup)

Use this after confirming B (`--init-random-points 120000`) is clearly better than A.

## Baseline (B)
```bash
SCENE=JAX_068
SRC=/home/m/EOGS-SFM/data/affine_models/${SCENE}
IMGS=/home/m/EOGS-SFM/data/${SCENE}/images
OUT=/home/m/EOGS-SFM/output

cd src/gaussiansplatting
python train.py \
  -s ${SRC} \
  --images ${IMGS} \
  --eval \
  -m ${OUT}/${SCENE,,}_ft_B_base \
  --sh_degree 0 \
  --iterations 10000 \
  --init-random-points 120000 \
  --init-random-points-bbox-scale 1.05 \
  --init-scale-ceiling 0.03
```

## T1: soften over-darkening / halo           最好
```bash
python train.py \
  -s ${SRC} \
  --images ${IMGS} \
  --eval \
  -m ${OUT}/${SCENE,,}_ft_T1 \
  --sh_degree 0 \
  --iterations 30000 \
  --init-random-points 120000 \
  --init-random-points-bbox-scale 1.05 \
  --init-scale-ceiling 0.025 \
  --init-opacity 0.02 \
  --opacity-lr 0.015
```

## T2: sharper boundaries (risk: floaters)
```bash
python train.py \
  -s ${SRC} \
  --images ${IMGS} \
  --eval \
  -m ${OUT}/${SCENE,,}_ft_T2 \
  --sh_degree 0 \
  --iterations 10000 \
  --init-random-points 120000 \
  --init-random-points-bbox-scale 1.05 \
  --init-scale-ceiling 0.02 \
  --init-opacity 0.025 \
  --opacity-lr 0.02
```

## T3: denoise dark regions (risk: slight blur)
```bash
python train.py \
  -s ${SRC} \
  --images ${IMGS} \
  --eval \
  -m ${OUT}/${SCENE,,}_ft_T3 \
  --sh_degree 0 \
  --iterations 10000 \
  --init-random-points 120000 \
  --init-random-points-bbox-scale 1.05 \
  --init-scale-ceiling 0.03 \
  --init-opacity 0.018 \
  --opacity-lr 0.012
```

## Compare
Render and compare B/T1/T2/T3 at iteration 7000 or 10000:
- building edge sharpness
- dark area cleanliness
- halos around high-contrast boundaries
- floaters in roads / vegetation

## Decision rule (when differences are small)
- If T2 only slightly improves edge sharpness but introduces any road/vegetation floaters, prefer T1.
- If T1/T2/T3 are visually very close, prefer T1 as the stable default.
- Promote to T2 only when boundary sharpness gain is clear in multiple views.

