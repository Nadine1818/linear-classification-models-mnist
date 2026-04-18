# Linear Classification Models on MNIST

This repository contains a single, end-to-end notebook project that implements and evaluates **linear classifiers on MNIST** (from-scratch and PyTorch-based), plus a **feedforward neural network** baseline.

The goal is to demonstrate understanding of:
- data preparation and train/val/test splitting
- manual gradient-based training loops (no `optimizer.step()` for the “manual” sections)
- early stopping and basic evaluation (accuracy + confusion matrices)

## What’s implemented (from `final.ipynb`)

### 1) Binary Logistic Regression (manual, digits 0 vs 1)

- Filters MNIST to only digits **0 and 1**.
- Uses parameters $W \in \mathbb{R}^{784\times 1}$ and $b \in \mathbb{R}$, with:
   - `sigmoid(x @ W + b)`
   - binary cross-entropy loss
- Trains with a **manual update** loop:
   - `learning_rate = 0.01`, `max_epochs = 100`
   - **early stopping** with `patience = 5` and improvement threshold `1e-4`
- Evaluates on test set with:
   - printed test accuracy
   - a **manual confusion matrix** visualization

### 2) Softmax Regression (manual, 10-class)

- Uses parameters $W \in \mathbb{R}^{784\times 10}$ and $b \in \mathbb{R}^{10}$.
- Implements:
   - numerically-stable `softmax`
   - cross-entropy loss
   - accuracy helper
- Trains with a **manual update** loop:
   - `learning_rate = 0.01`, `max_epochs = 100`, `patience = 5`
   - early stopping uses the same `best_val_loss - 1e-4` rule
- Evaluates with:
   - loss/accuracy curves
   - confusion matrix (heatmap)
   - **per-class accuracy** printout

### 3) Softmax Regression (PyTorch “built-in” wrapper)

- Defines `SoftmaxRegression(nn.Module)` using a single `nn.Linear(784, 10)` layer.
- Uses:
   - `criterion = nn.CrossEntropyLoss()` (applies softmax internally)
   - `optimizer = optim.SGD(..., lr=0.01)`
- Trains with early stopping:
   - tracks best validation loss (`best_val_loss - 1e-4`)
   - stops after `patience` epochs without improvement
- Evaluates with:
   - loss/accuracy curves
   - confusion matrix + per-class accuracy

### 4) Neural Network baseline (feedforward + trainer)

- Implements a configurable feedforward network (`NeuralNetwork(nn.Module)`):
   - `hidden_sizes: List[int]`, `activation: 'relu'|'tanh'`
   - `init_method: 'xavier'|'he'`
   - `dropout_rate` (default `0.2`)
- Trains via a `NeuralNetworkTrainer` that:
   - selects device automatically (`cuda` if available)
   - uses SGD + CrossEntropyLoss
   - includes early stopping (`patience = 5`, `min_delta = 0.001`) and restores the best model
- Saves weights to `best_nn_model.pth`.

## Data pipeline (reproducible)

The notebook loads MNIST via `torchvision.datasets.MNIST(root='./data', download=True)` and:
- normalizes pixels to `[0, 1]` by dividing by `255.0`
- flattens images from `28×28` to `784`

Splitting:
- Uses a **60% train / 20% validation / 20% test** split via `train_test_split(..., stratify=..., random_state=42)`.
- For multiclass softmax sections, it concatenates the MNIST train+test sets into one full pool before splitting.

## Results Summary (test accuracy)

Accuracies below are copied from the printed outputs in `final.ipynb`.

- Binary Logistic Regression (manual, digits 0 vs 1): **0.9929** (99.29%)
- Softmax Regression (manual, 10-class): **0.8771** (87.71%)
- Softmax Regression (PyTorch): **0.9166** (91.66%)
- Neural Network baseline: **98.89%** (0.9889)

## Repository contents

- `final.ipynb` — main notebook (implementations + experiments)
- `final.html`, `finalll.pdf` — exported reports
- `best_nn_model.pth` — saved weights for the neural network baseline
- `*.png` — saved plots (loss/accuracy curves, learning curves, etc.)

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook
```

Open `final.ipynb` and run cells top-to-bottom.

## Notes

- Downloaded MNIST files under `./data/` are intentionally not committed to git.
- `.venv/` is intentionally ignored.
- Outputs (plots/exports) should match when using the same seeds, but can vary slightly across environments.
