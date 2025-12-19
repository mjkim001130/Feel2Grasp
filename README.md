<div align="center">

# Feel2Grasp

### Tactile-Conditioned Offline RL for Re-grasping

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github)](https://mjkim001130.github.io/Feel2Grasp/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge)](LICENSE)

**Minjae Kim** · **Kyoungin Baik** · **Juhyung Kim**

*Yonsei University MEU6505*

<br>

<img src="docs/assets/images/teaser_1.png" width="45%" alt="Feel2Grasp Teaser 1"/>
<img src="docs/assets/images/teaser_2.png" width="45%" alt="Feel2Grasp Teaser 2"/>

<p><i>Teleop collection setup and front camera + tactile sensor streams</i></p>

</div>

---

## Overview

**Feel2Grasp** aims to learn precise grasping behaviors using **tactile sensors** and **offline reinforcement learning (IQL)**. Given an initial imperfect grasp, the robot learns to repeatedly adjust its grasp until it reaches a desired contact configuration.

### Key Features

- **Tactile-Driven Re-grasping**: Uses tactile sensor feedback to determine grasp success
- **Offline RL (IQL)**: Trains policies from demonstration data without online interaction
- **Visual Feature Extraction**: AutoEncoder for compact front camera image representations
- **Frame Stacking**: Temporal context through 4-frame stacking for better motion understanding

### What Matters

| Component | Role |
|-----------|------|
| **Vision** | Helps reach and stabilize around the object |
| **Tactile** | Specifies the desired contact configuration |
| **State Stacking** | Adds temporal context for reliable re-grasping |

---

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
├── docs/                   # Project page
└── lerobot/                # LeRobot library
```

---

## Installation

### 1. Create Environment

```bash
conda create -y -n feel2grasp python=3.10
conda activate feel2grasp
conda install ffmpeg -c conda-forge
```

### 2. Install LeRobot

**From Source (Recommended):**
```bash
cd lerobot
pip install -e ".[feetech]"
```

**From PyPI:**
```bash
pip install 'lerobot[feetech]'
```

### 3. SO101 Robot Setup

Follow the official guide: [SO-101 Setup](https://huggingface.co/docs/lerobot/so101)

---

## Usage

### Step 1: Collect Data with SO101 Robot

```bash
cd lerobot
python so101_record.py
```

> **Tip**: During data collection, rely only on camera and sensor input (not direct observation). This naturally leads to re-grasping motions that help the policy learn retry behaviors.

### Step 2: Train AutoEncoder

```bash
jupyter notebook notebooks/train_AE.ipynb
```

Output: `outputs/ae_out/encoder.pt`

### Step 3: Create Replay Buffer

```bash
jupyter notebook notebooks/Replay_buffer.ipynb
```

### Step 4: Train IQL Policy

**Recommended: Frame-Stacked IQL (4 frames)**

```bash
python scripts/train/train_iql_stacked.py \
    --data data/replay_buffer_iql_72d_stacked4.npz \
    --project Feel2Grasp-IQL \
    --run_name iql_stacked4 \
    --steps 5000000
```

**Basic IQL (Single Frame)**

```bash
python scripts/train/train_iql.py \
    --data data/replay_buffer_iql_72d.npz \
    --project Feel2Grasp-IQL \
    --run_name iql_experiment \
    --steps 3000000
```

<details>
<summary><b>Training Arguments</b></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | - | Path to replay buffer |
| `--batch_size` | 256 | Training batch size |
| `--steps` | 3M (basic) / 5M (stacked) | Total training steps |
| `--gamma` | 0.99 | Discount factor |
| `--tau_expectile` | 0.7 | Expectile for value function |
| `--beta` | 3.0 | Temperature for AWR |
| `--save_dir` | `./IQL_checkpoints` | Checkpoint save directory |
| `--stack_size` | 4 | Number of stacked frames (stacked only) |

</details>

### Step 5: Evaluate on Robot

```bash
python scripts/eval/so101_iql_eval_smooth.py
```

Before running, update the checkpoint paths:
```python
IQL_CHECKPOINT_PATH = "checkpoints/iql_step_xxx.pt"
ENCODER_CHECKPOINT_PATH = "outputs/ae_out/encoder.pt"
```

<details>
<summary><b>Evaluation Options</b></summary>

| Option | Description |
|--------|-------------|
| `NUM_EPISODES` | Number of evaluation episodes |
| `FPS` | Control frequency (default: 25) |
| `MAX_JOINT_SPEED` | Maximum joint speed for smooth motion |
| `SAVE_DATASET` | Whether to save evaluation data |

</details>

---

## Results

Our experiments show that **state stacking with tactile sensing** achieves the best performance:

| Stage | Configuration | Result |
|-------|---------------|--------|
| 1 | No stacking + Tactile | Failed to approach or grasp |
| 2 | No stacking + Vision only | Grasps but doesn't re-attempt |
| 3 | **4-frame stacking + Tactile** | **Re-grasps until success** |

For detailed results and videos, visit our [Project Page](https://mjkim001130.github.io/Feel2Grasp/).

---

## Requirements

- CUDA 11.8
- PyTorch 2.6.0
- Python 3.10

---

## Citation

```bibtex
@misc{feel2grasp2025,
  title  = {Feel2Grasp: Tactile-Conditioned Offline RL for Re-grasping},
  author = {Kim, Minjae and Baik, Kyoungin and Kim, Juhyung},
  year   = {2025},
  howpublished = {\url{https://github.com/mjkim001130/Feel2Grasp}},
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**[Project Page](https://mjkim001130.github.io/Feel2Grasp/)** · **[Issues](https://github.com/mjkim001130/Feel2Grasp/issues)**

</div>
