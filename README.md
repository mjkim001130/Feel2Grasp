# Feel2Grasp

**Feel2Grasp: Precise Grasp using Tactile Sensors with Offline RL**

This repository is for **Yonsei MEU6505**.

## Overview

Feel2Grasp aims to learn precise grasping behaviors using tactile sensors and offline reinforcement learning.
We also train an AutoEncoder to extract visual features from the front camera images and use them as part of the RL state.

## Project Structure

```
Feel2Grasp/
├── src/                    # Core source code
│   ├── models/             # Model definitions (reward_latent, etc.)
│   └── utils/              # Utilities (circle detection, buffer creation)
├── scripts/                # Training and evaluation scripts
│   ├── train/              # IQL training scripts
│   └── eval/               # SO101 evaluation scripts
├── notebooks/              # Jupyter notebooks (AE training, replay buffer)
├── data/                   # Data files (.npz, .parquet)
├── checkpoints/            # Model checkpoints
├── outputs/                # Training outputs (ae_out, ae_side_out)
└── lerobot/                # LeRobot library
```

## Installation

```bash
conda create -y -n env-name python=3.10
conda activate env-name
```

When using `conda`, install `ffmpeg` in your environment:

```bash
conda install ffmpeg -c conda-forge
```

### Install LeRobot

#### From Source

First, clone the repository and navigate into the directory:

```bash
cd lerobot
```

Then, install the library in editable mode. This is useful if you plan to contribute to the code.

```bash
pip install -e ".[feetech]"
```

### Installation from PyPI

**Core Library:**
Install the base package with:

```bash
pip install 'lerobot[feetech]'
```

### so101

Follow the official SO-101 setup guide: [setup](https://huggingface.co/docs/lerobot/so101)

---

### Train AutoEncoder

This project is based on:

- CUDA 11.8
- PyTorch 2.6.0

We train an AutoEncoder to extract visual features from the front camera images, which are used in RL states.

## Usage

### 1. Collect Data with SO101 Robot

Use LeRobot to collect demonstration data:

```bash
cd lerobot
python so101_record.py
```

### 2. Train AutoEncoder

Train the autoencoder to extract visual features from front camera images:

```bash
# Run the Jupyter notebook
jupyter notebook notebooks/train_AE.ipynb
```

The encoder will be saved to `outputs/ae_out/encoder.pt`.

### 3. Create Replay Buffer

Create a replay buffer from the collected dataset:

```bash
# Run the Jupyter notebook
jupyter notebook notebooks/Replay_buffer.ipynb
```

### 4. Train IQL Policy

Train the IQL (Implicit Q-Learning) policy using the replay buffer.

#### Recommended: Frame-Stacked IQL (4 frames)

We recommend using the **frame-stacked version** for better temporal understanding and avoid causal confusion:

```bash
python scripts/train/train_iql_stacked.py \
    --data data/replay_buffer_iql_72d_stacked4.npz \
    --project Feel2Grasp-IQL \
    --run_name iql_stacked4 \
    --steps 5000000
```

Frame stacking concatenates 4 consecutive observations (72D x 4 = 288D), which helps the policy capture motion dynamics.

#### Basic IQL (Single Frame)

For single-frame training:

```bash
python scripts/train/train_iql.py \
    --data data/replay_buffer_iql_72d.npz \
    --project Feel2Grasp-IQL \
    --run_name iql_experiment \
    --steps 3000000
```

**Training Options:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | - | Path to replay buffer |
| `--batch_size` | 256 | Training batch size |
| `--steps` | 3,000,000 (basic) / 5,000,000 (stacked) | Total training steps |
| `--gamma` | 0.99 | Discount factor |
| `--tau_expectile` | 0.7 | Expectile for value function |
| `--beta` | 3.0 | Temperature for AWR |
| `--save_dir` | `./IQL_checkpoints` | Checkpoint save directory |
| `--stack_size` | 4 | Number of stacked frames (stacked version only) |

### 5. Evaluate on Robot

Run the trained policy on the SO101 robot:

```bash
python scripts/eval/so101_iql_eval_smooth.py
```

```bash
python scripts/eval/so101_iql_eval_smooth.py
```

Before running, update the checkpoint paths in the script:
```python
IQL_CHECKPOINT_PATH = "checkpoints/iql_step_xxx.pt"
ENCODER_CHECKPOINT_PATH = "outputs/ae_out/encoder.pt"
```

**Evaluation Options (in script):**
- `NUM_EPISODES`: Number of evaluation episodes
- `FPS`: Control frequency (default: 25)
- `MAX_JOINT_SPEED`: Maximum joint speed for smooth motion
- `SAVE_DATASET`: Whether to save evaluation data
