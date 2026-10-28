#!/bin/bash
#SBATCH --account=pi-ekempton
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH --job-name=wasp94_live
#SBATCH --signal=B:TERM@180

set -euo pipefail
snapshot=${1:?Pass the frozen source directory}
output_root=${2:?Pass a new benchmark output directory}
runner_dir=${3:?Pass the directory containing run_wasp94.py}
seconds=${4:-1500}
seed=${5:-94}
case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) live=5000; update=5000; new_bound=50000; label=live5000_fixed ;;
    1) live=20000; update=5000; new_bound=50000; label=live20000_fixed ;;
    2) live=20000; update=20000; new_bound=200000; label=live20000_coupled ;;
    *) echo 'Use task 0 (5k), 1 (20k, fixed updates), or 2 (20k, scaled updates)' >&2; exit 2 ;;
esac
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset PYTHONPATH
module load cuda/12.9
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
exec /home/tfairnington/miniconda3/envs/platon_env/bin/python -u \
    "${runner_dir}/run_wasp94.py" --snapshot "$snapshot" \
    --output "${output_root}/${label}_seed${seed}" --backend jax \
    --n-live "$live" --n-update "$update" --n-like-new-bound "$new_bound" \
    --seed "$seed" --seconds "$seconds" --max-calls 40000000
