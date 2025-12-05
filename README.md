# Feel2Grasp

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

Following the SO-101 [setup](https://huggingface.co/docs/lerobot/so101)

---

### Train AutoEncoder

This project based on `cuda 11.8`, `torch==2.6.0`

We extract visual feature from front camera images, using to RL states.