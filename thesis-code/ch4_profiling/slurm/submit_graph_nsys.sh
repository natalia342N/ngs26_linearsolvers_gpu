#!/bin/bash
#SBATCH --job-name=ngscuda_nsys
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

ml --force purge
ml load ASC/2023.06
ml load buildenv/default-foss-2023a
ml load CMake/3.26.3-GCCcore-12.3.0 CUDA/12.9.0 SciPy-bundle/2023.07-gfbf-2023a occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

WORKING_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV="$WORKING_DIR/ngs"
PREFIX="$WORKING_DIR/install"

source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# CUDA runtime libs visible (cusparse/cublas)
if [ -n "${EBROOTCUDA:-}" ] && [ -d "$EBROOTCUDA/lib64" ]; then
  export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
fi

# Use THIS install prefix
PYDIR=$(find "$PREFIX" -maxdepth 6 -type d \( -name site-packages -o -name dist-packages \) | head -n 1)
if [ -z "${PYDIR:-}" ]; then
  echo "ERROR: Could not find site-packages under $PREFIX"
  exit 1
fi
export PYTHONPATH="$PYDIR:${PYTHONPATH:-}"
export PATH="$PREFIX/bin:$PATH"

echo "WORKING_DIR=$WORKING_DIR"
echo "VENV=$VENV"
echo "PREFIX=$PREFIX"
echo "PYDIR=$PYDIR"
echo

which python
python -V
nvidia-smi
echo

# --- nsys output base name (nsys will add .qdrep, sometimes also .sqlite) ---
OUTBASE="$WORKING_DIR/nsys_ngscuda_${SLURM_JOB_ID}"
echo "nsys output base: $OUTBASE"
echo

# Profile: CUDA + OS runtime + NVTX (NVTX ranges show up if you installed nvtx)
# --force-overwrite true so reruns overwrite if same JOBID isn't possible anyway
nsys profile \
  -t cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o "$OUTBASE" \
  python "$WORKING_DIR/test_graph_capture.py"

echo
echo "Generated profile files:"
ls -lh "${OUTBASE}"* || true

