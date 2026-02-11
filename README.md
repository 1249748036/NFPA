# NFPA: Next-Frame Prediction Attack for AI Watermark Removal

Official implementation of **"The Future Unmarked: Watermark Removal in AI-Generated Images via Next-Frame Prediction"** (NeurIPS 2025).

## Overview

We present the core components of NFPA (Next-Frame Prediction Attack). The attack leverages zero-shot next-frame prediction with pretrained Text-to-Image (T2I) diffusion models to remove AI watermarks from generated images, without requiring any knowledge of the watermarking scheme.

- The attack pipeline uses a pretrained Stable Diffusion model (e.g., Stable Diffusion 2.1 base) to perform zero-shot next-frame prediction.
- Watermarked images are used as targets for this attack.

## Repository Structure

```
NFPA-main/
├── nfp_main.ipynb          # Main attack pipeline (model loading, attack execution, evaluation)
├── utils.py                # Stable Diffusion pipeline with next-frame prediction support
├── watermarks/
│   └── stable_watermark.obj  # Pre-generated Tree-Ring watermark pattern (for reproducibility)
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/1249748036/NFPA.git
cd NFPA
```

### 2. Create environment and install dependencies

```bash
conda create -n nfpa python=3.10 -y
conda activate nfpa
pip install -r requirements.txt
```

### 3. Prepare COCO dataset (for Tree-Ring watermark evaluation)

Download the [COCO 2017 Validation set](https://cocodataset.org/#download) and place it under `./data/`:


You can also modify the dataset paths directly in `nfp_main.ipynb` (Cell 1) to point to your local COCO dataset location.

## Usage

Open `nfp_main.ipynb` and run the cells sequentially:

1. **Cell 0** — Initialize the NFP attack pipeline (loads Stable Diffusion 2.1 base)
2. **Cell 1** — Initialize the Tree-Ring watermark system with COCO dataset
3. **Cell 2** — Define the NFP attack function and configure parameters
4. **Cell 3** — Generate watermarked images from COCO captions
5. **Cell 4** — Watermark detection before attack (baseline)
6. **Cell 5** — Execute NFP attack on watermarked images
7. **Cell 6** — Watermark detection after attack
8. **Cell 7** — Final summary of TPR@FPR results
9. **Cell 8** — (Optional) Visualize comparison of original / watermarked / attacked images

### Configuration

Key parameters can be adjusted in the notebook:

| Parameter | Default | Description |
|---|---|---|
| `device` | `"cuda"` | GPU device (auto-detected) |
| `NUM_IMAGES` | `1000` | Number of images to process |
| `NFP_INFERENCE_STEPS` | `10` | DDIM inversion/denoising steps |
| `NFP_XY` | `40` | Motion field search range for warping |

## Requirements

- Python 3.10
- PyTorch >= 2.2.2
- CUDA-compatible GPU (recommended: >= 24GB VRAM)

See `requirements.txt` for the full list of dependencies.

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{qiufuture,
  title={The Future Unmarked: Watermark Removal in AI-Generated Images via Next-Frame Prediction},
  author={Qiu, Huming and Wang, Zhaoxiang and Zhang, Mi and Zhang, Xiaohan and You, Xiaoyu and Yang, Min},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
