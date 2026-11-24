# FGAesQ Food Photography

This project adapts the pretrained [FGAesQ](https://github.com/yzc-ippl/FG-IAA)
image-aesthetic model to food photography from the Gourmet Photography Dataset
(GPD). The current phase only covers environment setup, pretrained inference,
dataset indexing, and feature extraction. Domain adaptation and training are
not implemented.

## Repository structure

- `FG-IAA/`: original FGAesQ implementation, kept as close as possible to the
  authors' code.
- `src/fgaesq_project/`: project-specific model loading, inference, and GPD
  indexing helpers.
- `scripts/`: setup checks, image inference, feature verification, and manifest
  generation.
- `checkpoints/`: local pretrained weights; ignored by Git.
- `data/gpd/`: local GPD files; ignored by Git.
- `data/samples/`: local smoke-test images; ignored by Git.
- `results/`: generated outputs; ignored by Git.

## Windows setup

FGAesQ's released requirements currently pin PyTorch 2.6.0 and CLIP. Python
3.12 is recommended for those wheels. From the repository root in PowerShell:

```powershell
py install 3.12  # only if Python 3.12 is not already installed
Remove-Item -Recurse -Force .venv  # only when replacing an incompatible venv
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install setuptools==75.8.0
python -m pip install --no-build-isolation -r FG-IAA\FGAesQ_Inference\requirements.txt
```

The pinned CLIP source package still imports `pkg_resources` during its build;
the setuptools pin and `--no-build-isolation` keep that official dependency
install reproducible without modifying the research code.

The requirements use the standard PyTorch package index. If `nvidia-smi`
reports a working NVIDIA driver, verify the installed wheel with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

On a CPU-only machine, the same code automatically uses CPU when
`--device auto` is selected. If the standard PyTorch install does not provide
a compatible CUDA wheel for the installed driver, install the matching wheel
from the [official PyTorch selector](https://pytorch.org/get-started/locally/)
instead of changing the research code.

Validated on this Windows machine: Python `3.12.10`, PyTorch
`2.7.1+cu128`, torchvision `0.22.1+cu128`, CUDA runtime `12.8`, and an NVIDIA
RTX 5060 Laptop GPU (`sm_120`). The authors' PyTorch `2.6.0` pin was tested
first, but its CUDA 12.4 wheel does not contain kernels for `sm_120`; the
CUDA override below is therefore required for this GPU:

```powershell
python -m pip install --force-reinstall `
  --index-url https://download.pytorch.org/whl/cu128 `
  torch==2.7.1 torchvision==0.22.1
python -m pip install --force-reinstall `
  --index-url https://pypi.org/simple `
  numpy==2.0.1 pillow==11.2.1 setuptools==75.8.0
```

## Pretrained checkpoint

Download `FGAesQ.pt` from the official
[Hugging Face model](https://huggingface.co/yzc002/FGAesQ) and save it as:

```text
checkpoints\FGAesQ.pt
```

The official model also loads OpenAI CLIP ViT-B/16. On first construction,
CLIP may download its visual backbone into its normal cache. Check the
checkpoint and environment with:

```powershell
python scripts\check_setup.py
```

## Inference

Score one image or all supported images below a directory:

```powershell
python scripts\run_inference.py `
  --input data\samples `
  --output results\sample_scores.csv `
  --device auto
```

The CSV contains the image path, predicted score (on the released 1--10
distribution expectation), and an explicit error column. An image that fails
to decode or preprocess is not silently assigned a score.

## GPD download and manifest

The official [GPA repository](https://github.com/Openning07/GPA) identifies
GPD as a non-commercial research/educational dataset. It reports 24,000
resized images:

| Partition | Positive | Negative | Total |
|---|---:|---:|---:|
| Train | 11,791 | 9,809 | 21,600 |
| Test | 1,311 | 1,089 | 2,400 |

Download the official **Positive Partition (Resized)** and **Negative
Partition (Resized)** using the Google Drive links in that repository, or use
its listed Jianguoyun mirrors. Extract them under `data\gpd\`, preserving
positive/negative directory names. The links are:

- Positive: `https://drive.google.com/open?id=1hQUtgNBXijFP1ET82AazlhxQ6Rz38DgY`
- Negative: `https://drive.google.com/open?id=1vlPalmsrkHRUJ3razt4tR-V3w7FJGrqi`

If Google Drive requires an interactive login or download confirmation,
download the two archives in a browser, accept the dataset access terms, and
extract them manually. Do not commit the archives or images.

Create a reusable manifest after extraction:

```powershell
python scripts\index_gpd.py `
  --root data\gpd `
  --output data\gpd_manifest.csv
```

The index records relative paths, positive/negative labels, any `train`,
`test`, or `validation` directory split, SHA-256 hashes, invalid images, and
duplicates. If the downloaded layout does not contain split directories, the
manifest marks the split as `unspecified`; it does not invent a test split.

## Feature extraction verification

The official implementation uses `feature_dim = 512` and supports
`model(..., flag=True)`, returning features and scores. Verify both on local
images:

```powershell
python scripts\verify_features.py `
  --input data\samples `
  --checkpoint checkpoints\FGAesQ.pt `
  --device auto
```

The command prints the finite aesthetic score and feature shape for every
image. It does not train a food-specific head.

## Current status and next phase

Implemented: project-local paths, checkpoint loading with compatibility
checks, CPU/CUDA device selection, explicit-error CSV inference, GPD
manifesting, and 512-dimensional feature verification.

Not implemented: baseline evaluation protocol, food-specific scoring-head
training, domain adaptation, or use of the GPD test set for model selection.
The next phase is baseline evaluation followed by a separately designed
domain-adaptation experiment.
