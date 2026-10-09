# Running MP3D lesion detection on this machine

This documents the working local setup: a single-GPU (RTX 3050, 6 GB) Windows
run of the MICCAI 2020 MP3D universal lesion detector on real DeepLesion CT
slices.

## What this model detects

DeepLesion is a single-class detection task: the model localises **lesions**
(tumours / enlarged lymph nodes / other measurable abnormalities that a
radiologist bookmarked) in axial CT slices. It draws boxes and gives each a
confidence; it does **not** name a specific disease. The published metric is
FROC — sensitivity at a fixed number of false positives per image.

## Quick start on another Windows PC

Download the repository (Code > Download ZIP and extract it, or `git clone`)
and double-click `run_project.bat`. The first run needs internet and about
8 GB of free disk space, and takes 15-30 minutes: it installs the Visual C++
runtime if missing, then uv, Python 3.10 and the exact package versions from
`requirements-lock.txt` into `.tools/` and `.venv/`, and downloads the final
model, the annotations and the CT slices around the 32 built-in samples from
the release. Then the web app starts and the browser opens. Later runs skip
straight to the app and work offline.

It runs on an NVIDIA GPU when one works with this CUDA 11.3 build and falls
back to the CPU otherwise (a few seconds per image); set `MP3D_DEVICE=cpu` to
force the CPU. If port 5000 is taken the app uses the next free port.

## Downloads: trained models and data

Everything too large for the repository is attached to the
[v1.0 release](https://github.com/kowshickj-git/MP3D-LESION/releases/tag/v1.0):
the trained detectors (every saved epoch of both runs), the pretrained
backbone, and DeepLesion parts 01-04 with their annotations. The web app and
evaluation only need the trained checkpoints:

```bash
R=https://github.com/kowshickj-git/MP3D-LESION/releases/download/v1.0
mkdir -p work_dirs/mp3d_lesion_hires work_dirs/mp3d_lesion checkpoints
curl -L -o work_dirs/mp3d_lesion_hires/epoch_7.pth $R/mp3d_lesion_hires_epoch_7.pth  # final model (run 2)
curl -L -o work_dirs/mp3d_lesion/latest.pth $R/mp3d_lesion_epoch_12.pth              # run 1
curl -L -o checkpoints/mp3d63-d720bda1.pth $R/mp3d63-d720bda1.pth                     # backbone, for training
```

The DeepLesion zips are over GitHub's 2 GB file limit, so each is split into
nine pieces of about 480 MB that are joined back into NIH's original archive:

```bash
mkdir -p data/DeepLesion/zips
for n in 01 02 03 04; do
  for p in $(seq -f %03g 9); do curl -L -o Images_png_$n.zip.$p $R/Images_png_$n.zip.$p; done
  cat Images_png_$n.zip.0* > data/DeepLesion/zips/Images_png_$n.zip && rm Images_png_$n.zip.0*
done
(cd data/DeepLesion/zips && curl -sL $R/SHA256SUMS.txt | sha256sum -c --ignore-missing)
curl -L -o DeepLesion_annotation.zip $R/DeepLesion_annotation.zip
python -m zipfile -e DeepLesion_annotation.zip data/DeepLesion   # annotation/ and DL_info.csv
python tools/prepare_deeplesion.py --root data/DeepLesion
```

On Windows without Git Bash, join with `copy /b Images_png_01.zip.0* Images_png_01.zip`
or open the `.001` piece in 7-Zip.

## 1. Environment

MMDetection 2.25 needs `mmcv-full` in `[1.3.17, 1.6.0]`, whose prebuilt
Windows wheels stop at CPython 3.10 — hence a 3.10 venv rather than the
system 3.11.

```bash
uv venv --python 3.10 .venv
UV_CACHE_DIR=/d/uv-cache VIRTUAL_ENV=$PWD/.venv uv pip install \
    torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
    --index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match
UV_CACHE_DIR=/d/uv-cache VIRTUAL_ENV=$PWD/.venv uv pip install \
    https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/mmcv_full-1.6.0-cp310-cp310-win_amd64.whl
UV_CACHE_DIR=/d/uv-cache VIRTUAL_ENV=$PWD/.venv uv pip install \
    "setuptools==65.5.1" numpy==1.23.5 opencv-python matplotlib pycocotools six \
    terminaltables addict yapf==0.32.0 pandas xlsxwriter SimpleITK scipy einops \
    timm requests
UV_CACHE_DIR=/d/uv-cache VIRTUAL_ENV=$PWD/.venv uv pip install -e . --no-deps --no-build-isolation
```

Notes:
* `uv`'s cache must live on D: — C: has under 4 GB free and the torch wheel
  extraction fails with "not enough space" partway through.
* setuptools must be < 81; newer releases drop `pkg_resources`, which
  `torch.utils.cpp_extension` still imports.
* `requirements-lock.txt` pins the exact versions of the working venv,
  including flask for the web app, and installs in one step:
  `uv pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy unsafe-best-match`.

Verify:

```bash
.venv/Scripts/python.exe -c "import torch,mmcv,mmdet; from mmcv.ops import nms; \
print(torch.__version__, torch.cuda.get_device_name(0), mmcv.__version__, mmdet.__version__)"
```

## 2. Data

Annotations (COCO-style, converted from DeepLesion by the MVP-Net authors):

```bash
mkdir -p data/DeepLesion/annotation
for f in train val test; do
  curl -sL -o data/DeepLesion/annotation/deeplesion_$f.json \
    https://raw.githubusercontent.com/urmagicsmine/MVP-Net/master/data/DeepLesion/annotation/deeplesion_$f.json
done
```

CT images — the full release is 56 zips / ~225 GB. Grab as many parts as you
want into `data/DeepLesion/zips/` (each ~4.3 GB, ~80 patients, ~500 annotated
slices). Part 01:

```bash
curl -L -C - -o data/DeepLesion/zips/Images_png_01.zip \
  https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip
```

The remaining URLs are in NIH's `batch_download_zips.py`. Parts 01-04, the
ones used here, are also on the release (see Downloads). Then extract and
build annotation subsets restricted to the studies actually on disk:

```bash
python tools/prepare_deeplesion.py --root data/DeepLesion
```

This writes `deeplesion_{train,val,test}_local.json`. Re-run it after adding
more zips; partially-downloaded archives are skipped.

## 3. Pretrained weights

```bash
curl -L "https://drive.usercontent.google.com/download?id=1jKdJQ83vZrvOT4N_iZeRvhDuyKWiboqi&export=download&confirm=t" \
  -o checkpoints/mp3d63-d720bda1.pth
```

The same file is on the release as `mp3d63-d720bda1.pth`.

This is the COCO-supervised MP3D63 model from the paper, **not** a trained
lesion detector — it is an mmdet-1.x state_dict with `backbone.` / `neck.` /
`rpn_head.` / `bbox_head.` prefixes and 81 COCO classes.
`configs/deeplesion/mp3d_groupconv_local.py` therefore warm-starts the
backbone, FPN and RPN head from it via `init_cfg=dict(..., prefix=...)` and
trains the lesion head from scratch. The four `groupconv_*` layers (the
paper's 3D->2D context conversion) are also new, as the released model was
trained with `conversion_type='center_crop'`.

## 4. Train

```bash
.venv/Scripts/python.exe tools/train.py \
    configs/deeplesion/mp3d_groupconv_local.py \
    --work-dir work_dirs/mp3d_lesion
```

The stock `mp3d_groupconv.py` targets 8 GPUs at 448-576px. On 6 GB that
overflows VRAM into host memory and runs at ~7.5 s/image. The local config
uses 320-384px with the stem and layer1 frozen, which fits in ~4 GB and runs
at ~0.43 s/image — about 17x faster.

## 5. Evaluate (COCO mAP + the paper's FROC)

```bash
.venv/Scripts/python.exe tools/test.py \
    configs/deeplesion/mp3d_groupconv_local.py \
    work_dirs/mp3d_lesion/latest.pth --eval bbox
```

## 6. Detect lesions on CT slices and save pictures

```bash
.venv/Scripts/python.exe tools/lesion_demo.py \
    configs/deeplesion/mp3d_groupconv_local.py work_dirs/mp3d_lesion/latest.pth \
    --ann data/DeepLesion/annotation/deeplesion_test_local.json \
    -n 12 --score-thr 0.5 --out work_dirs/demo
```

Green boxes are the radiologist ground truth, red are the model's detections
with confidences. `--img 000001_01_01/109.png --slice-intv 5.0` runs a single
named slice instead.

## Fixes needed to make the repo run

* `mmdet/datasets/pipelines/image_io_3DCE.py` split slice paths on `os.sep`,
  which is `\` on Windows, while the annotations use `/`. Every 9-slice load
  raised `ValueError: invalid literal for int()`. Now splits on either.
* `mmcv/utils/env.py` (`collect_env`) aborts `tools/train.py` on a machine
  with no MSVC installed; the compiler probe is now non-fatal. This is a
  site-packages patch, so reapply it if the venv is rebuilt.

## Results

Run 1 — `mp3d_groupconv_local.py`, 12 epochs, 1010 train slices (DeepLesion
parts 01-02), 320-384px, batch 2, ~35 min on one RTX 3050. Evaluated with
`tools/test.py` on 433 held-out test slices:

| FP / image | Sensitivity |
| ---------- | ----------- |
| 0.5        | 7.59%       |
| 1          | 9.60%       |
| 2          | 12.50%      |
| 4          | 17.19%      |
| ~99 (max)  | 53.79%      |

Mean FROC 11.72; COCO bbox mAP 0.014, mAP@50 0.040, mAP_large 0.187.

Validation progression (168 slices) shows the run was still improving when it
stopped, with the usual jump at the learning-rate drops:

| epoch | mAP@50 | Mean FROC | Sens@4FP |
| ----- | ------ | --------- | -------- |
| 4     | 0.023  | -         | -        |
| 8     | 0.058  | 3.22      | 4.68%    |
| 12    | 0.065  | 11.84     | 14.62%   |

For reference the paper reports ~85% sensitivity @ 4FP. That model is trained
on all 22,496 training slices on 8 GPUs at 448-576px; this one sees 4.5% of
that data on a single 6 GB laptop GPU, so the gap is a compute/data budget
difference rather than a pipeline difference.

`mAP_large` (0.187) being 8x `mAP_medium` (0.023) says the input downscaling
is what hurts most - small lesions are the ones being missed. Run 2
(`mp3d_groupconv_local_hires.py`) therefore doubles the data to 2033 slices
and raises the input to 448-512px, trading batch size for resolution, warm
started from run 1.

### Run 2 (final) - `mp3d_groupconv_local_hires.py`

7 epochs, 2033 train slices (parts 01-04), 448-512px, batch 1, warm started
from run 1. ~2h30m on the RTX 3050. Same 433 held-out test slices:

| FP / image | Run 1 (384px, 1010 slices) | Run 2 (512px, 2033 slices) |
| ---------- | -------------------------- | -------------------------- |
| 0.5        | 7.59%                      | **28.57%**                 |
| 1          | 9.60%                      | **35.94%**                 |
| 2          | 12.50%                     | **46.65%**                 |
| 4          | 17.19%                     | **56.92%**                 |
| ~99 (max)  | 53.79%                     | **81.03%**                 |
| Mean FROC  | 11.72                      | **42.02**                  |
| mAP        | 0.014                      | **0.099**                  |
| mAP@50     | 0.040                      | **0.224**                  |

A 3.6x gain in mean FROC. The `mAP_large` >> `mAP_medium` gap in run 1
correctly identified input resolution as the binding constraint: restoring it
(and doubling the data) lifted `mAP_medium` from 0.023 to 0.137 on validation.

Validation progression during run 2 (267 slices):

| epoch | mAP@50 | Mean FROC |
| ----- | ------ | --------- |
| 2     | 0.162  | 32.41     |
| 4     | 0.211  | 36.30     |
| 6     | 0.226  | 37.78     |

Note on reading the logs: the FROC block is emitted with `print()` while the
mAP goes through the mmcv logger, so when stdout is redirected to a file the
FROC lines lag their epoch by a flush. `tools/test.py` flushes on exit, so
prefer it for authoritative numbers.

For reference the paper reports ~85% sensitivity @ 4FP using all 22,496
training slices on 8 GPUs. Run 2 sees 9% of that data on one 6 GB laptop GPU
and reaches 56.9%.

## Web front end

```bash
.venv/Scripts/python.exe webapp/app.py
# then open http://127.0.0.1:5000
```

It loads `work_dirs/mp3d_lesion_hires/epoch_7.pth` (see Downloads).

Drag a CT slice onto the drop zone (or click one of the eight built-in
samples) and the trained detector boxes the lesions it finds, with a
confidence for each. The threshold slider has presets matching the measured
operating points from the FROC curve, so "2 FP" really does mean about two
false positives per image.

**The 3D-context caveat.** MP3D consumes a 9-slice stack, not one image, so a
single uploaded slice is not by itself enough. The backend handles this in two
ways and tells you which one it used:

* *real* (green note) - the filename identifies a study in
  `data/DeepLesion`, e.g. `000016_01_01__008.png`, so the 9 genuine
  neighbouring slices are loaded. This is the mode the benchmark numbers were
  measured in. Files taken from `samples/` keep their names and land here.
* *replicated* (amber note) - any other image. The slice is repeated 9 times.
  The model still runs and still finds obvious lesions, but with no real
  through-plane context it produces more false positives; on the sample tested
  the same lesion scored 88% instead of 90% but picked up an extra FP.

16-bit DeepLesion PNGs are windowed with the training window
(`[-1024, 1050]` HU after the +32768 offset); 8-bit images are assumed to be
already windowed.


Endpoints: `GET /api/samples`, `POST /api/detect` (multipart `image`,
`score_thr`, `show_gt`) returning JSON with the boxes and an annotated PNG as
a data URI.

Two things to know if you edit it: Flask caches compiled templates when
`debug=False`, so `TEMPLATES_AUTO_RELOAD` is enabled to make HTML edits show
up without a restart (the model reload costs ~30s); and any CSS class that
sets `display` will override the browser's `[hidden]` rule, which is why the
stylesheet declares `[hidden]{display:none !important}`.

## Per-region pages

The front end is organised by the eight coarse body regions DeepLesion labels:

    /                overview, all regions
    /r/lung  /r/mediastinum  /r/liver  /r/kidney
    /r/abdomen  /r/pelvis  /r/bone  /r/soft-tissue

Each region page carries three lesion cases spanning the size range plus one
lesion-free slice from the same anatomy, and reports per detection:
confidence, long/short axis in millimetres, a size band, and whether the box
matches the radiologist's at IoU >= 0.5.

Each region page carries four written sections - what the lesions in that
region actually are, why radiologists measure them, how they present on CT and
why that makes them easy or hard to detect, and what this system does with
them - plus that region's measured hit rate. The text lives in the `REGIONS`
list in `webapp/app.py`, one dict per region with `pathology`, `why`, `ct` and
`perf` keys, so editing the copy needs no template changes.

**The region is still not predicted.** MP3D is single-class - every box it
draws is just "Lesion". The region assigned to each page is DeepLesion ground
truth from `Coarse_lesion_type` in `DL_info.csv`, not a model output. The
model supplies location, confidence and size; the clinical framing on each
page is context for the reader, not something inferred from the pixels.

`DL_info.csv` is not in the repo and the NIH Box link for it is dead; the
build script expects it at `data/DeepLesion/DL_info.csv`. It is inside
`DeepLesion_annotation.zip` on the release, or can be fetched from the
HuggingFace mirror:

```bash
curl -L -o data/DeepLesion/DL_info.csv   https://huggingface.co/datasets/farrell236/DeepLesion/resolve/main/DL_info.csv
.venv/Scripts/python.exe tools/build_region_samples.py
```

Only 718 of our local lesions carry a coarse type (the other 2087 are
unlabelled in DL_info), which is enough for all eight regions - bone is
thinnest at 15.

### Sizing

Long axis = the longer side of the predicted box x the slice's `Spacing_mm_px_`
from DL_info. Bands are `sub-centimetre` (<10 mm, below the RECIST measurable
threshold), `small` (10-20), `moderate` (20-30) and `large` (>=30). These are
millimetre thresholds, **not** clinical grades or stages. Boxes tend to run a
little larger than the radiologist's calliper measurement - on the lung sample
above the model reports 37.3 mm where the radiologist measured 30 mm.

### Measured per-region behaviour

Every sample run through `/api/detect` at the 2 FP threshold (0.155):

| region | lesions | matched @ IoU 0.5 | sensitivity | FPs on the normal slice |
| ------ | ------- | ----------------- | ----------- | ----------------------- |
| lung | 3 | 3 | 100% | 1 |
| mediastinum | 3 | 3 | 100% | 0 |
| pelvis | 3 | 2 | 67% | 2 |
| soft tissue | 3 | 2 | 67% | 0 |
| abdomen | 3 | 1 | 33% | 1 |
| bone | 3 | 1 | 33% | 1 |
| kidney | 3 | 1 | 33% | 1 |
| liver | 4 | 0 | 0% | 1 |
| **total** | **25** | **13** | **52%** | |

52% on this handful is consistent with the 56.9% @ 4 FP measured over the full
433-slice test set. Liver going 0/4 is the standout weakness and tracks the
training data - only 64 liver lesions were present locally, against 107 lung
and 143 mediastinum.

The `normal__` slices are negative controls, not certified-healthy images:
DeepLesion only marks lesions a radiologist chose to bookmark, so a detection
there may be a real but unbookmarked finding rather than a false positive.
They sit at least 12 slices from any bookmark in studies with few bookmarks.
Raising the threshold to the 1 FP preset clears most of them.
