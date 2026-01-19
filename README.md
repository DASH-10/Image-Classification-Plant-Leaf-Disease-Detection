# Plant Leaf Disease Classification (PlantVillage)

**Author:** Omar A.M Issa (Student Number: 220212901)

## Overview
This project builds an end-to-end pipeline for classifying plant leaf diseases from images. It covers data ingestion and preprocessing, a classical machine learning baseline (HOG + SVM), a transfer-learning model (ResNet-18), and evaluation with visual diagnostics like confusion matrices, precision-recall curves, and Grad-CAM. The main notebook runs the full flow and saves artifacts (figures, tables, models) to `outputs`.

## Why this project
Plant diseases reduce yield and quality, and early detection is critical in agriculture. This project explores two complementary approaches:
- A lightweight, interpretable baseline (handcrafted features + SVM) that is fast to train and serves as a sanity check.
- A deep learning model (ResNet-18) that captures richer visual patterns and typically achieves higher accuracy.

The goal is to compare these approaches on a standard dataset, document the tradeoffs, and provide a reproducible reference pipeline.

## Dataset
The project uses the PlantVillage dataset, which contains labeled images of healthy and diseased crop leaves. In this repo, the processed dataset has:
- **24 classes**
- **Train:** 19,862 images
- **Val:** 4,256 images
- **Test:** 4,257 images
- **Total:** 28,375 images

One class is `Background_without_leaves`, and the rest are crop/disease or crop/healthy categories.

### Why this dataset
- It is a widely used benchmark for plant disease classification, making results easy to compare.
- The labels are clear and consistent, which helps isolate model performance from annotation noise.
- It is large enough to support both classical ML baselines and transfer learning without being prohibitively huge.

### Where to get it
The pipeline supports two sources:
1) **TensorFlow Datasets (preferred)**  
   The notebook will try `plant_village` or `plant_village/plantvillage` and download automatically if available.

2) **Kaggle (fallback)**  
   Dataset name: `emmarex/plantdisease`  
   The code will run:
   ```
   kaggle datasets download -d emmarex/plantdisease
   

If you already downloaded PlantVillage manually, place the folder under `data/raw/` with class subfolders.

## Project structure
```
project_leaf_disease/
  notebook/
    leaf_disease_classification.ipynb   # End-to-end pipeline and experiments
  src/
    config.py                           # Paths and global hyperparameters
    data.py                             # Dataset download, splitting, loaders
    features.py                         # HOG + color histogram feature extraction
    models.py                           # SVM pipeline and ResNet-18 builder
    train.py                            # Training loops, sweeps, and search
    eval.py                             # Metrics, predictions, PR curves
    viz.py                              # Plotting utilities and Grad-CAM
    utils_seed.py                       # Reproducibility and env info
  data/
    raw/                                # Downloaded dataset (TFDS or Kaggle)
    processed/                          # Train/val/test splits + splits.json
  outputs/
    figures/                            # Plots and visual diagnostics
    tables/                             # CSV summaries
    models/                             # Saved checkpoints
    logs/                               # Optional logs
  report/                               # Report sources and PDF output
  requirements.txt                      # Python dependencies
  README.md
```

## Requirements and setup
- Python 3.10+ recommended
- Jupyter Notebook or JupyterLab
- Key libraries: PyTorch, torchvision, scikit-learn, scikit-image, numpy, pandas, matplotlib, tensorflow-datasets, kaggle
- Full list in `requirements.txt`

### Setup (venv)
Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
```
Windows:
```bash
.venv\Scripts\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Install:
```bash
python -m pip install -r requirements.txt
```

### Common errors and fixes
- `ModuleNotFoundError`: the notebook kernel is not using `.venv`. Re-select the kernel and restart.
- `Import 'torch' could not be resolved`: install CPU-safe PyTorch with `python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
- `Kaggle API error`: place `kaggle.json` under `C:\Users\<user>\.kaggle\kaggle.json` or set `KAGGLE_USERNAME` / `KAGGLE_KEY`.

## How to run
1) Open the notebook:
   ```bash
   jupyter notebook notebook/leaf_disease_classification.ipynb
   ```
2) Run all cells top-to-bottom. Outputs will be saved under `outputs/`.

## Feature caching
The classical HOG pipeline caches extracted features under `outputs/cache/`. Cache keys include
the image path list + mtimes, image size, and HOG params. On reruns, features are loaded from
disk using memory mapping for lower RAM use. Any unreadable images are logged to
`outputs/cache/bad_images.txt` and skipped to keep alignment consistent.

To clear the cache, delete the `outputs/cache/` folder.

## Faster SVM search (subset)
To keep the classical baseline usable on medium PCs, the SVM hyperparameter search runs on a
configurable subset of the training set by default (stratified). After the best params are found,
the final model is refit on the full training data.

Adjust the subset size via `run_hog_svm_search(..., search_subset_size=8000)` or disable it by
passing `search_subset_size=0`.

## Limitations and downfalls
- **High memory usage:** HOG + color histogram extraction builds large feature matrices for tens of thousands of images. This can consume significant RAM and slow down or freeze the notebook on low-memory machines.
- **Heavy training workload:** ResNet-18 training and sweeps can take a long time on CPU and may use large amounts of GPU memory on CUDA systems, affecting responsiveness.
- **Disk usage:** The pipeline copies images into `data/processed/` and stores outputs in `outputs/`, which increases storage usage.

If the notebook feels slow, reduce `BATCH_SIZE`, lower image size, or skip sweeps in the notebook.

## Reproducibility
- Fixed seeds in `src/utils_seed.py` for Python, NumPy, PyTorch, and DataLoader workers.
- Splits are stratified and saved to `data/processed/splits.json`.
- Environment details are printed in the notebook header via `get_env_info()`.

## Output artifacts
- `outputs/figures/`: class distribution, sample grids, confusion matrices, PR curves, Grad-CAM, misclassifications
- `outputs/tables/`: summary tables (CSV)
- `outputs/models/`: saved model checkpoints
- `outputs/logs/`: optional logs

## Notes
- The notebook includes a hyperparameter sweep for SVM and a small LR/WD sweep for the DL model.
- Fine-tuning vs frozen-backbone ablation is included to satisfy the optimization and ablation requirements.
