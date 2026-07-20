#!/bin/bash
#SBATCH --job-name=diamond_test
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

ml --force purge
ml load ASC/2023.06
ml load CMake/3.26.3-GCCcore-12.3.0 CUDA/12.9.0
ml load SciPy-bundle/2023.07-gfbf-2023a occt/7.8.0-GCCcore-12.3.0
ml load buildenv/default-foss-2023a
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

WORKING_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "$WORKING_DIR/ngs/bin/activate"

PREFIX="$WORKING_DIR/install"
PYDIR="$PREFIX/lib/python3.11/site-packages"
NUMPY_DIR=/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/amd/zen4/software/SciPy-bundle/2023.07-gfbf-2023a/lib/python3.11/site-packages
export PYTHONPATH="$PYDIR:$NUMPY_DIR"
export PATH="$PREFIX/bin:$PATH"

echo "=== WHILE graph ==="
python "$WORKING_DIR/test_devcg2.py"

echo "=== DIAMOND graph ==="
USE_DIAMOND_GRAPH=1 python "$WORKING_DIR/test_devcg2.py"
