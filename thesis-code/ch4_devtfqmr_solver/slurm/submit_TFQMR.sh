#!/bin/bash
#SBATCH --job-name=convection_gmres_nsys
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ---------------- pick script ----------------
# Usage:
#   sbatch submit_convection_nsys.sh
#   sbatch submit_convection_nsys.sh convectionGMRes.py
# SCRIPT_REL="${1:-convectionTFQMR_graph.py}"
SCRIPT_REL="${1:-convection_opgraph.py}"
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
SCRIPT="$WORKING_DIR/$SCRIPT_REL"

# Prefer a filesystem with quota headroom for nsys output:
# - use $DATA if available, else fall back to working dir
OUTROOT="${DATA:-$WORKING_DIR}"
OUTDIR="$OUTROOT/nsys"
mkdir -p "$OUTDIR"

# ---------------- Activate venv ----------------
if [ ! -f "$VENV/bin/activate" ]; then
  echo "ERROR: venv not found at $VENV"
  echo "Expected: $VENV/bin/activate"
  exit 1
fi
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# ---------------- Threading sanity (avoids 384 threads surprises) ----------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ---------------- CUDA runtime libs visible (cublas/cusparse) ----------------
if [ -n "${EBROOTCUDA:-}" ] && [ -d "$EBROOTCUDA/lib64" ]; then
  export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
fi

# ---------------- Use THIS install prefix (ngsolve/netgen built under $PREFIX) ----------------
PYDIR=$(find "$PREFIX" -maxdepth 6 -type d \( -name site-packages -o -name dist-packages \) | head -n 1)
if [ -z "${PYDIR:-}" ]; then
  echo "ERROR: Could not find site-packages under $PREFIX"
  exit 1
fi
export PYTHONPATH="$PYDIR"
export PATH="$PREFIX/bin:$PATH"

# ---------------- Script exists? ----------------
if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: Script not found: $SCRIPT"
  echo "WORKING_DIR contents:"
  ls -lah "$WORKING_DIR" || true
  exit 1
fi

# ---------------- Diagnostics ----------------
echo "WORKING_DIR=$WORKING_DIR"
echo "VENV=$VENV"
echo "PREFIX=$PREFIX"
echo "PYDIR=$PYDIR"
echo "SCRIPT=$SCRIPT"
echo "OUTDIR=$OUTDIR"
echo
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo
which python
python -V
echo
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
OUTBASE="$OUTDIR/nsys_convection_${SLURM_JOB_ID}"
echo "nsys output base: $OUTBASE"
echo

# Keep nsys options minimal & reliable for CUDA trace collection.
# Use srun so the profiled process is correctly attached to the SLURM allocation/cgroups.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
  nsys profile \
    -o "$OUTBASE" \
    -t cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite=true \
    python "$SCRIPT"

echo
echo "Generated profile files:"
ls -lh "${OUTBASE}"* || true

echo
echo "Quick stats check:"
# These will work if a .qdrep was produced; otherwise they fail gracefully.
nsys stats --report cudaapisum "${OUTBASE}.qdrep" 2>/dev/null || true
nsys stats --report cudakernsum "${OUTBASE}.qdrep" 2>/dev/null || true

echo
python - <<'PY'
import ngsolve
print("NGSolve version:", getattr(ngsolve, "__version__", "unknown"))
print("ngsolve:", getattr(ngsolve, "__file__", "unknown"))
try:
    import ngsolve.ngscuda as ngscuda
    print("ngscuda:", getattr(ngscuda, "__file__", "unknown"))
    print("Has CudaGraph:", hasattr(ngscuda, "CudaGraph"))
    print("Has DeviceSynchronize:", hasattr(ngscuda, "DeviceSynchronize"))
except Exception as e:
    print("WARNING: Could not import ngsolve.ngscuda:", e)
PY

echo
echo "Job finished at $(date)"
