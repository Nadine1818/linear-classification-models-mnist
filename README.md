# Linear Classification Models (MNIST)

This project implements and analyzes linear classification models on **MNIST** (plus a neural network baseline) using **PyTorch**.

## Contents

- `final.ipynb` — main notebook
- `final.html`, `finalll.pdf` — exported reports
- `best_nn_model.pth` — saved model weights
- `*.png` — training/analysis plots

## Data

This repository is configured to **not commit the downloaded MNIST raw files**.

The notebook uses `torchvision.datasets` and can download MNIST automatically.
If you need the files locally, keep them under:

- `data/MNIST/`

## Run locally

1. Create/activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run:

   ```bash
   jupyter notebook
   ```

Open `final.ipynb`.
