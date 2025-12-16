# Feel2Grasp

**Feel2Grasp: Precise Grasp using Tactile Sensors with Offline RL**

This repository is for **Yonsei MEU6505**.

## Overview

Feel2Grasp aims to learn precise grasping behaviors using tactile sensors and offline reinforcement learning.
We also train an AutoEncoder to extract visual features from the front camera images and use them as part of the RL state.

## Installation


```bash
conda create -y -n env-name python=3.10
conda activate env-name
```

When using `conda`, install `ffmpeg` in your environment:

```bash
conda install ffmpeg -c conda-forge
```

### Install LeRobot 🤗

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