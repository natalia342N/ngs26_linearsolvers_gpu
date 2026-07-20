#!/bin/bash
#SBATCH --job-name=ngscuda_nsys_graph_norm
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ---------------- Modules ----------------
ml --force purge
ml load ASC/2023.06
ml load buildenv/default-foss-2023a
ml load CMake/3.26.3-GCCcore-12.3.0
ml load CUDA/12.9.0
ml load SciPy-bundle/2023.07-gfbf-2023a
ml load occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

# ---------------- Paths ----------------
WORKING_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV="$WORKING_DIR/ngs"
PREFIX="$WORKING_DIR/install"

source "$VENV/bin/activate"

# Avoid accidental shadowing
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# Make CUDA runtime libs visible (cublas/cusparse)
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

# ---------------- Diagnostics ----------------
echo "WORKING_DIR=$WORKING_DIR"
echo "VENV=$VENV"
echo "PREFIX=$PREFIX"
echo "PYDIR=$PYDIR"
echo
which python
python -V
nvidia-smi
echo

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo

# ---------------- nsys sanity ----------------
if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not found in PATH. (Try loading a different CUDA module or a dedicated Nsight Systems module.)"
  exit 1
fi
nsys --version || true
echo

# ---------------- Profile run ----------------
OUTBASE="$WORKING_DIR/nsys_graph_norm_${SLURM_JOB_ID}"
echo "nsys output base: $OUTBASE"
echo

# Recommended for CUDA Graph debugging:
# - cuda: kernels, memcpys, graph launches
# - nvtx: if you add NVTX ranges later
# - osrt: useful to see CPU runtime overhead
nsys profile \
  -t cuda,nvtx,osrt \
  --stats=true \
  --force-overwrite=true \
  -o "$OUTBASE" \
  python "$WORKING_DIR/test_graph_norm.py"

echo
echo "Generated profile files:"
ls -lh "${OUTBASE}"* || true

echo
python - <<'PY'
import ngsolve, ngsolve.ngscuda as ngscuda
print("NGSolve version:", getattr(ngsolve, "__version__", "unknown"))
print("Has CudaGraph:", hasattr(ngscuda, "CudaGraph"))
print("ngscuda:", getattr(ngscuda, "__file__", "unknown"))
PY

echo
echo "Job finished at $(date)"
