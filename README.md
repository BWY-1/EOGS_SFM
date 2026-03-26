# EOGS

List of tools and scripts developped while prepping a Gaussian-splatting version of EO-NeRF

## How to install

* Create a conda environment
```bash
conda create -n eogs python=3.10
conda activate eogs
```
* Install pytorch and torchvision
```bash
pip install torch torchvision          
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```
* Install packages in `requirements.txt` 
```bash
pip install -r requirements.txt
```
* Install 3DGS CUDA kernels
```bash
pip install src/gaussiansplatting/submodules/diff-gaussian-rasterization
TORCH_CUDA_ARCH_LIST="Pascal;Volta;Turing;Ampere" python -m pip install -e src/gaussiansplatting/submodules/diff-gaussian-rasterization --no-build-isolation
pip install src/gaussiansplatting/submodules/simple-knn
```

## How to create dataset
Download the dataset from the GitHub Release page: [dataset_v01](https://github.com/mezzelfo/EOGS/releases/download/dataset_v01/data.zip)

Extract the dataset in the `data` folder (`unzip -q data.zip -d data`). The structure should look like this:
```
EOGS/
├── data/
│   ├── images/
│   │   ├── JAX_004/
│   │   │   ├── JAX_004_006_RGB.tif
│   │   │   ├── JAX_004_007_RGB.json
│   │   │   ...
|   │   ├── JAX_068/
|   │   ...
│   ├── rpcs/
│   │   ├── JAX_004/
│   │   │   ├── JAX_004_006_RGB.json
│   │   │   ├── JAX_004_007_RGB.json
│   │   │   ...
│   │   │   ├── test.txt
│   │   │   └── train.txt
|   │   ├── JAX_068/
|   │   ...
│   ├── truth/
│   │   ├── JAX_004/
│   │   │   ├── JAX_004_CLS.tif
│   │   │   ├── JAX_004_DSM.txt (optional)
│   │   │   └── JAX_004_DSM.tif
|   │   ├── JAX_068/
|   │   ...
│   ├── README.md/
```

Copy the dataset DFC2019 and root_dir in a _datasets_ folder. Then run the following commands to create the affine approximations of the camera models:
```bash
python scripts/dataset_creation/to_affine.py --scene_name JAX_004
python scripts/dataset_creation/to_affine.py --scene_name JAX_068
python scripts/dataset_creation/to_affine.py --scene_name JAX_214
python scripts/dataset_creation/to_affine.py --scene_name JAX_260
python scripts/dataset_creation/to_affine.py --scene_name IARPA_001
python scripts/dataset_creation/to_affine.py --scene_name IARPA_002
python scripts/dataset_creation/to_affine.py --scene_name IARPA_003
```


If you want to initialize training from an external sparse point cloud (e.g. SatelliteSfM), convert it to the normalized affine world frame and save it as `data/affine_models/{SCENE}/points3d.ply`:
```bash
python scripts/dataset_creation/convert_satellitesfm_ply_to_eogs.py \
  --input-ply /home/m/EOGS-SFM/data/JAX_068/point_cloud.ply\
  --affine-models-json data/affine_models/JAX_068/affine_models.json \
  --output-ply data/affine_models/JAX_068/points3d.ply \
  --input-coord auto \
  --dry-run-report \
  --crop-to-scene-bounds
```

For SatelliteSfM ENU outputs, prefer explicit ENU conversion:
```bash
python scripts/dataset_creation/convert_satellitesfm_ply_to_eogs.py \
  --input-ply /home/m/EOGS-SFM/data/JAX_068/point_cloud.ply\
  --affine-models-json data/affine_models/JAX_068/affine_models.json \
  --output-ply data/affine_models/JAX_068/points3d.ply \
  --input-coord enu \
  --enu-observer-json data/JAX_068/enu_observer_latlonalt.json
```
Important safety note for ENU conversion: this chain is most reliable on small scenes (roughly <1km², no UTM-zone crossing).  
For larger areas, direct geodetic integration is recommended; ENU mode now enforces safety checks by default and can be overridden with `--allow-enu-unsafe`.


`--input-coord auto` now falls back to `bbox_fit` by default when both `utm` and `normalized` do not match the affine bounds.  
If you prefer strict behavior, set `--auto-fallback error`.
For a more detailed auto-mode diagnosis (bbox + camera-consistency scoring), add `--dry-run-report`.

If conversion reports `Output points: 0`, first retry without cropping (`--crop-to-scene-bounds`) to verify coordinate alignment, then tune `--input-coord` and `--bounds-margin`.

You can diagnose coordinate mismatch before conversion:
```bash
python scripts/dataset_creation/check_affine_pointcloud_alignment.py \
  --input-ply /home/m/EOGS-SFM/data/affine_models/JAX_068/points3d.ply\
  --affine-models-json data/affine_models/JAX_068/affine_models.json
```

If you are checking a converted output from `--input-coord bbox_fit`, you can pass `--input-coord bbox_fit` to the checker (it will be interpreted as normalized/world coordinates).
Note: the default converted filename is `points3d.ply` (lowercase `d`).
The checker also reports camera consistency ratio stats across cameras (uv in `[-1,1]` and altitude within bounds), which is useful to detect cases where bbox overlap looks okay but rendering is still background-only.

Important: use the `affine_models.json` generated under `data/affine_models/{SCENE}/`, not a raw metadata file in another folder.
If both `utm` and `normalized` diagnostics fail (in-bounds ratio near 0), try:
```bash
python scripts/dataset_creation/convert_satellitesfm_ply_to_eogs.py \
  --input-ply /path/to/satellitesfm_sparse.ply \
  --affine-models-json data/affine_models/JAX_004/affine_models.json \
  --output-ply data/affine_models/JAX_004/points3d.ply \
  --input-coord bbox_fit \
  --bbox-fit-pca-align \
  --bbox-fit-anisotropic
```
Tip: for isotropic bbox fitting, the converter now defaults to `--bbox-fit-isotropic-axes xy` to avoid altitude range dominating the scale.
If converter warns that `z_span` is much larger than affine bbox Z span, prefer `--bbox-fit-anisotropic` (or lower `--bbox-fit-fill-ratio`).

cd src/gaussiansplatting
python train.py \
  -s /home/m/EOGS-SFM/data/affine_models/JAX_068 \
  --images /home/m/EOGS-SFM/data/JAX_068/images \
  --eval \
  -m /home/m/EOGS-SFM/output/jax068_debug \
  --sh_degree 0 \
  --iterations 10000


## How to reproduce the results
Run the following command to reproduce the results of Table 1 in the [paper](https://arxiv.org/pdf/2412.13047) (be aware that different initial random seeds in `src/gaussiansplatting/utils/general_utils.py/safe_state` will lead to potentially different results):
```bash
bash train.sh reproduceMain
```

# FAQ

> [!TIP]
> if `No module named 'torch'` when install: `submodules/diff-gaussian-rasterization`, `pip install --upgrade setuptools wheel packaging`
 
> [!TIP]
> if `KeyError: 'centerofscene_ECEF` while running the code: regenerate the camera models (see [dataset creation](#how-to-create-dataset))

> [!TIP]
> When using uv: if `No module named 'torch'` when install: `submodules/diff-gaussian-rasterization`, `--no-build-isolation` (recommended by the latest uv version)


dsm_name=$(ls /home/m/EOGS-SFM/output/jax068_debug_safe2/test_opNone/ours_7000/dsm/ | sort -V | tail -n 1)

python /home/m/EOGS-SFM/scripts/eval/eval_dsm.py \
  --pred-dsm-path /home/m/EOGS-SFM/output/jax068_debug_safe2/test_opNone/ours_7000/dsm/${dsm_name} \
  --gt-dir /home/m/EOGS-SFM/data/truth/JAX_068 \
  --out-dir /home/m/EOGS-SFM/output/jax068_debug_safe2\
  --aoi-id JAX_068


python scripts/dataset_creation/convert_satellitesfm_ply_to_eogs.py \
  --input-ply /home/m/EOGS-SFM/data/JAX_068/point_cloud.ply \
  --affine-models-json /home/m/EOGS-SFM/data/affine_models/JAX_068/affine_models.json \
  --output-ply /home/m/EOGS-SFM/data/affine_models/JAX_068/points3d_enu.ply \
  --input-coord enu \
  --enu-observer-json /path/to/enu_observer_latlonalt.json \
  --dry-run-report