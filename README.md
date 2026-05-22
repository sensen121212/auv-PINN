# AUV trajectory prediction / PINN project

This repository contains code for AUV trajectory data preparation, corruption/audit utilities, and trajectory prediction experiments using PINN and neural network baselines.

## Paths

- Local project path: `D:\数学建模\total_matlab\会议\AUV_dataset`
- Server project path: `/disk/baosen/project/AUV_dataset`
- Docker runtime path: `/workspace/AUV_dataset`
- Docker container: `baosen_pinn_dev`

## Main Directories

- `PINN/`: AUV trajectory prediction models, training/evaluation scripts, baselines, and benchmark runners.
- Project root scripts:
  - `prepare_data.py`: data preparation utilities.
  - `prepare_batch_data.py`: batch data preparation utilities.
  - `step1_data_explore_and_corrupt.py`: data exploration and corruption workflow.
- `output/`, `trajectory_dataset/`, date-named folders, checkpoints, logs, figures, and model exports are generated data/results and are not tracked by Git.

## Setup

```bash
cd /workspace/AUV_dataset
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Docker Usage

```bash
docker start baosen_pinn_dev
docker exec -it baosen_pinn_dev bash
cd /workspace/AUV_dataset
```

## Common Commands

```bash
python prepare_data.py
python prepare_batch_data.py
python step1_data_explore_and_corrupt.py

cd PINN
python train.py
python evaluate.py
python train_baseline.py
python -m benchmark.run_benchmark
python -m benchmark.collect_results
```

## Git Sync

```bash
git status
git add .
git commit -m "describe your change"
git push
```

Only commit source code, lightweight configuration, and documentation. Do not commit raw datasets, generated datasets, checkpoints, exported model weights, training logs, generated figures, benchmark outputs, or large intermediate files.
