# AUV PINN

This repository contains code for AUV trajectory prediction experiments using a physics-informed neural network (PINN) and related baselines.

## Paths

- Local project path: `D:\数学建模\total_matlab\会议\AUV_dataset\PINN`
- Server project path: `/disk/baosen/project/AUV_dataset/PINN`
- Docker runtime path: `/workspace/AUV_dataset/PINN`
- Docker container: `baosen_pinn_dev`

## Setup

```bash
cd /workspace/AUV_dataset/PINN
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the server, run commands inside the container:

```bash
docker exec -it baosen_pinn_dev bash
cd /workspace/AUV_dataset/PINN
```

## Common Commands

```bash
python train.py
python evaluate.py
python train_baseline.py
python -m benchmark.run_benchmark
python -m benchmark.collect_results
```

## Repository Policy

Only commit source code, lightweight configuration, and documentation. Do not commit the full AUV dataset, checkpoints, exported model weights, training logs, generated figures, or benchmark outputs.
