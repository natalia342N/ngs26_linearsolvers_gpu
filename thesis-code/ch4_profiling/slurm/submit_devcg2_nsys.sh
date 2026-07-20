#!/bin/bash
#SBATCH --job-name=devcg2_nsys
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
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
VENV="$WORKING_DIR/ngs"
PREFIX="$WORKING_DIR/install"

source "$VENV/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NGS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

if [ -n "${EBROOTCUDA:-}" ] && [ -d "$EBROOTCUDA/lib64" ]; then
  export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
fi

PYDIR="$(find "$PREFIX" -maxdepth 6 -type d \( -name site-packages -o -name dist-packages \) | head -n 1 || true)"
export PYTHONPATH="$PYDIR:${PYTHONPATH:-}"
export PATH="$PREFIX/bin:$PATH"

OUTBASE="$WORKING_DIR/devcg2_nsys-${SLURM_JOB_ID}"
SCRIPT="$WORKING_DIR/test_devcg2.py"

HELP_TXT="$(nsys profile --help 2>&1 || true)"

NSYS_ARGS=(
  profile
  -t cuda,nvtx,osrt
  --cuda-event-trace=false
  --stats=true
  --force-overwrite=true
  -o "$OUTBASE"
)

fi
fi
if echo "$HELP_TXT" | grep -qi -- '--sample'; then
  NSYS_ARGS+=(--sample=none)
fi
if echo "$HELP_TXT" | grep -qi -- '--cpuctxsw'; then
  NSYS_ARGS+=(--cpuctxsw=none)
fi

echo "Running: nsys ${NSYS_ARGS[*]} python $SCRIPT"
nsys "${NSYS_ARGS[@]}" python "$SCRIPT"

echo "Generated profile files:"
ls -lh "${OUTBASE}"* || true

echo "Job finished at $(date)"
