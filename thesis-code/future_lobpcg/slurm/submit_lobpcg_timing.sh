#!/bin/bash
#SBATCH --job-name=lobpcg_timing
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

ml --force purge
ml load ASC/2023.06
ml load buildenv/default-foss-2023a
ml load CMake/3.26.3-GCCcore-12.3.0
ml load CUDA/12.9.0
ml load SciPy-bundle/2023.07-gfbf-2023a
ml load occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

WORKING_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PREFIX="$WORKING_DIR/install"
VENV="$WORKING_DIR/ngs"

source "$VENV/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NGS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

PYDIR="$(find "$PREFIX" -maxdepth 6 -type d \( -name site-packages -o -name dist-packages \) | head -n 1 || true)"
NUMPY_DIR=/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen4/software/SciPy-bundle/2023.07-gfbf-2023a/lib/python3.11/site-packages
export PYTHONPATH="$PYDIR:$NUMPY_DIR:${PYTHONPATH:-}"
export PATH="$PREFIX/bin:$PATH"

if [ -n "${EBROOTCUDA:-}" ] && [ -d "$EBROOTCUDA/lib64" ]; then
  export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "=== LOBPCG CPU vs GPU timing ==="
python "$WORKING_DIR/test_lobpcg_timing.py"
echo "=== Done ==="
