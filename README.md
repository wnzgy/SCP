# Adversarial Training with SCP and SPP

This repository contains the implementation of **SCP** and **SPP**, two novel adversarial attack and training methods designed to improve the robustness of deep neural networks.

## Supported Features

### Datasets
- CIFAR-10
- CIFAR-100
- MNIST
- Tiny-ImageNet

## Requirements

The project relies on PyTorch and utilizes `cupy` for efficient GPU-accelerated linear algebra operations required by SCP.

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training

Use `main.py` to train models using standard adversarial training or the proposed SCP/SPP methods.

**Arguments:**
- `--data`: Dataset name (`cifar10`, `mnist`, `tiny-imagenet`, `cifar100`).
- `--model`: Model architecture (e.g., `resnet18`, `vit_b_16`).
- `--attack`: The adversarial method and steps, format `method#steps` (e.g., `scp#5`, `spp#5`, `pgd#5`).
- `--epochs`: Number of training epochs.
- `--results_dir`: Directory to save logs and models.

**Example: Training ResNet-18 on CIFAR-10 using SCP (5 steps)**
```bash
python main.py --data cifar10 --model resnet18 --attack scp#5 --epochs 100
```

**Example: Training ResNet18 on Tiny-ImageNet using SPP**
```bash
python main.py --data tiny-imagenet --model resnet18 --attack spp#5 --batch_size 64
```

### 2. Evaluation

Use `evaluation.py` to evaluate the robustness of trained models against standard attacks (PGD, CW, AutoAttack).

**Arguments:**
- `--model_path`: Path to the `.pth` weight file.
- `--model_arch`: Model architecture used during training.
- `--dataset`: Dataset used.
- `--attacks`: Comma-separated list of attacks to test (e.g., `cw,auto`).

**Example:**
```bash
python evaluation.py \
  --model_path ./results/scp/cifar10/models/resnet18_cifar10_scp#5_best.pth \
  --model_arch resnet18 \
  --dataset cifar10 \
  --attacks cw,auto
```

## Project Structure

*   `main.py`: Main entry point for training models.
*   `evaluation.py`: Script for evaluating model robustness.
*   `scp.py`: Implementation of the SCP attack class.
*   `spp.py`: Implementation of the SPP attack class.
*   `myutil.py`: Utility functions for logging, seeding, and configuration management.
*   `utils/`: Helper modules.

## Results

Training logs and checkpoints are saved in the `results/` directory by default.
*   **Logs:** Training history and metrics (`.yaml` and `.log`).
*   **Models:** Best and final model weights (`.pth`).
*   **Weights:** Optimizer and training state checkpoints.

