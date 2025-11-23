# Infinite Mixture Prototypes (Updated Version)

## _Written by Lolézio Viora Marquet_

## Goal

The below provided code uses Python Version 2.7.13 and PyTorch version 0.3.1, which are both long out of support.

To be compatible with [R4RR](https://github.com/Giotto-maker/r4rr/tree/main), they need to be updated to Python 3.8.20 and PyTorch 1.13 (CUDA 11.7). This is what is implemented in this repository.

## Instructions to run experiments with updated version

1. Create a new Conda virtual environment (we use Conda as `pip` no longer supports Torch 1.xx)

    - Install Anaconda or Miniconda ([instructions source](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation))
        - On Windows PowerShell:
        - `Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile ".\Downloads\Miniconda3-latest-Windows-x86_64.exe"`
        - Run the `".\Downloads\Miniconda3-latest-Windows-x86_64.exe"` file and finish installation with the wizard
        - Open the Anaconda Prompt and run
        - `conda init powershell`
        - to add access to `conda` in PowerShell

    - Create and activate the virtual environment
        - `conda create -n imp-py38 python=3.8.20 -y`
        - `conda activate imp-py38`

2. Install the correct Python and PyTorch versions
    - CUDA version if you have a GPU:
        - `conda install pytorch==1.13.0 torchvision==0.14.0 cudatoolkit=11.7 -c pytorch -c conda-forge -y`
    - CPU-only version:
        - `conda install pytorch==1.13.0 torchvision==0.14.0 cpuonly -c pytorch -y`
    - Check installation:
        - `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
        - Should show `1.13.0 False` for CPU only and `1.13.0 True` for CUDA

3. Install requirements
    - `pip install -r requirements.txt`

---

# Infinite Mixture Prototypes

## Kelsey Allen, Evan Shelhamer, Hanul Shin, Josh Tenenbaum

## Abstract

We propose infinite mixture prototypes to adaptively represent both simple and complex data distributions for few-shot learning. Infinite mixture prototypes combine deep representation learning with Bayesian nonparametrics, representing each class by a set of clusters, unlike existing prototypical methods that represent each class by a single cluster. By inferring the number of clusters, infinite mixture prototypes interpolate between nearest neighbor and prototypical representations in a learned feature space, which improves accuracy and robustness in the few-shot regime. We show the importance of adaptive capacity for capturing complex data distributions such as super-classes (like alphabets in character recognition), with 10-25% absolute accuracy improvements over prototypical networks, while still maintaining or improving accuracy on standard few-shot learning benchmarks. By clustering labeled and unlabeled data with the same rule, infinite mixture prototypes achieve state-of-the-art semi-supervised accuracy, and can perform purely unsupervised clustering, unlike existing fully- and semi-supervised prototypical methods.

## Link to paper

http://proceedings.mlr.press/v97/allen19b.html

## Code

This repository is adapted from https://github.com/renmengye/few-shot-ssl-public for PyTorch 0.3.1

### Installation

We use Python 2.7.13. Other versions may work with some modifications.
To install requirements:

```bash
pip install -r requirements.txt
```

### Usage Examples

`submit_omniglot.sh` provides example usage of the main file.

We also have submission scripts for running code on a slurm cluster.
Please refer to `submit_all_models.sh` and `submit_super.sh` for examples.
