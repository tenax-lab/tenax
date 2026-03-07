# Installation

## Prerequisites

- Python 3.11 or 3.12
- A working JAX installation (CPU or GPU)

## Install from source

> **Note:** The PyPI package (`tenax-tn`) is not yet available. Install from source using the instructions below.

### With uv (recommended)

```bash
git clone https://github.com/tenax-lab/tenax.git
cd tenax
uv sync --all-extras --dev
```

### With pip

```bash
git clone https://github.com/tenax-lab/tenax.git
cd tenax
pip install -e ".[dev]"
```

## Hardware acceleration

Tenax uses JAX as its backend. To enable GPU or TPU acceleration,
install the appropriate JAX variant **before** installing Tenax:

```bash
# NVIDIA GPU (CUDA 13, recommended)
pip install -U "jax[cuda13]"

# NVIDIA GPU (CUDA 12)
pip install -U "jax[cuda12]"

# NVIDIA GPU with locally installed CUDA
pip install -U "jax[cuda12-local]"
pip install -U "jax[cuda13-local]"

# Google Cloud TPU
pip install -U "jax[tpu]"

# Apple Silicon GPU (macOS only, experimental)
pip install jax-metal
```

For AMD ROCm GPUs, install JAX with ROCm support separately following
[AMD's installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html),
then install Tenax on top.

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for the latest accelerator options.

## Building the documentation

```bash
uv sync --extra docs
cd docs && uv run make html
```

The built site will be in `docs/_build/html/`.

## Float64 precision

Tenax defaults to `float64` for all tensors and algorithms. Importing
`tenax` automatically enables JAX 64-bit mode via
`jax.config.update("jax_enable_x64", True)`.

If you import JAX *before* `tenax` and create arrays in that window, they
will still be `float32`. To avoid surprises, either import `tenax` first or
enable x64 manually:

```python
import jax
jax.config.update("jax_enable_x64", True)  # before any array creation

import tenax  # also calls the same update
```

See {doc}`gotchas` for more details on float64 behaviour.

## Verifying the installation

```python
import tenax
print(tenax.__version__)
```
