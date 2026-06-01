#!/bin/bash
#SBATCH --job-name=bench566
#SBATCH --account=acd109079
#SBATCH --partition=normal            # H100:8 nodes, 2-day limit. See header for H200.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1             # single GPU; the benchmark uses one device
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=benchmarks/results/bench566_%j.out
#SBATCH --error=benchmarks/results/bench566_%j.err
#
# #566 / #569 symmetric iPEPS CTM-AD batching benchmark (TENAX_BATCH_BLOCKSPARSE).
# Measures the off-vs-on crossover bond dimension D on a datacenter GPU.
#
# Cluster: H100/H200 (8 GPUs/node, ~1.9TB RAM). Partitions (from `sinfo`):
#     dev      H100:8   1:00:00     (quick smoke / compile-wall probe)
#     normal   H100:8   2-00:00:00  (default below)
#     4nodes   H100:8   1-00:00:00
#     normal2  H200:8   2-00:00:00  (H200)
#
# Submit on H100 (defaults above):
#     sbatch benchmarks/bench566_h100_h200.sbatch
#
# Submit on H200 -- override partition + gres at submit time (no edit needed):
#     sbatch --partition=normal2 --gres=gpu:H200:1 benchmarks/bench566_h100_h200.sbatch
#
# Quick smoke on the 1h dev partition:
#     sbatch --partition=dev --time=01:00:00 benchmarks/bench566_h100_h200.sbatch

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p benchmarks/results

# jax[cuda12] wheels bundle their own CUDA runtime, so no `module load cuda` is
# needed for the GPU path; loading a system CUDA can in fact shadow the bundled
# libs. The cuda/12.6 module is available if you ever switch to cuda12-local.
export PATH="$HOME/.local/bin:$PATH"

# Pin JAX to the single GPU SLURM gave us and keep XLA from grabbing all VRAM.
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

GPU_TAG="${SLURM_JOB_PARTITION:-gpu}"
STAMP="${SLURM_JOB_ID:-local}"

echo "=== node: $(hostname)  partition: ${SLURM_JOB_PARTITION:-?}  job: ${STAMP} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Ensure the GPU JAX stack is present in the project venv.
uv sync --extra cuda12

# f64 -- the realistic iPEPS AD regime (the decision metric).
uv run --extra cuda12 python examples/bench_symmetric_ad_batching_566.py \
    --D 2 3 4 6 8 --chi-factor 3 --reps 5 \
    --json "benchmarks/results/bench566_${GPU_TAG}_f64_${STAMP}.json"

# f32 -- latency-bound probe; on a datacenter GPU it stresses the small-kernel
# regime the batching path targets.
uv run --extra cuda12 python examples/bench_symmetric_ad_batching_566.py \
    --no-x64 --D 2 3 4 6 8 --chi-factor 3 --reps 5 \
    --json "benchmarks/results/bench566_${GPU_TAG}_f32_${STAMP}.json"

echo "=== done; JSON + .out in benchmarks/results/ -- attach to issue #569 ==="
