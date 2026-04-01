#!/bin/bash

# 定义公共参数（不用改）
SOURCE_PATH="/home/m/EOGS-SFM/data/affine_models/JAX_068_uniform"
IMAGES="/home/m/EOGS-SFM/data/images/JAX_068"
GT_DIR="/home/m/EOGS-SFM/data/truth/JAX_068"
AOI_ID="JAX_068"
ITERATIONS=30000

# 基础训练参数
BASE_ARGS="--sh_degree 0 --init-scale-ceiling 0.025 --init-opacity 0.02 --opacity-lr 0.015 --w-L-chromaticity 0.08 --iterstart-L-global-color 100 --w-L-global-color 0.15"

# ===================== 自动跑 4 组实验 =====================
for RP in 0 40000 120000 200000 
do
    MODEL_PATH="/home/m/EOGS-SFM/output/JAX_068_rp$RP"
    
    echo "=================================================="
    echo "  正在运行：init-random-points = $RP"
    echo "  输出路径：$MODEL_PATH"
    echo "=================================================="

    python scripts/eval/full_pipeline_eval.py \
      --source-path "$SOURCE_PATH" \
      --images "$IMAGES" \
      --model-path "$MODEL_PATH" \
      --gt-dir "$GT_DIR" \
      --aoi-id "$AOI_ID" \
      --iterations $ITERATIONS \
      --train-extra-args "$BASE_ARGS --init-random-points $RP"

    echo -e "\n✅ 完成 random points = $RP\n"
done

echo "🎉 所有 4 组实验全部跑完！"
