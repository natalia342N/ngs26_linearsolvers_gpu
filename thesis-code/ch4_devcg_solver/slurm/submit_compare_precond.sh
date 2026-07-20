#!/bin/bash
#SBATCH --job-name=compare_precond
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ml --force purge
ml load ASC/2025.06
ml load CUDA/12.9.0
ml load SciPy-bundle/2025.06-gfbf-2025a
ml load occt/7.9.1-GCCcore-14.2.0

WORKING_DIR="/home/nt51825/starting_with_ngscuda"
PREFIX="$WORKING_DIR/testbuild_2025_install"

GCC14_LIB=/cvmfs/software.eessi.io/versions/2025.06/software/linux/x86_64/amd/zen4/software/GCCcore/14.2.0/lib64
OCCT_LIB=/cvmfs/software.eessi.io/versions/2025.06/software/linux/x86_64/amd/zen4/software/occt/7.9.1-GCCcore-14.2.0/lib
CUDA_LIB="${CUDA_HOME}/lib64"
export LD_LIBRARY_PATH=${GCC14_LIB}:${OCCT_LIB}:${CUDA_LIB}:${LD_LIBRARY_PATH:-}

PYDIR="$PREFIX/lib/python3.13/site-packages"
NUMPY_DIR=$(python3 -c "import numpy, os; print(os.path.dirname(os.path.dirname(numpy.__file__)))")
export PYTHONPATH="$PYDIR:$NUMPY_DIR"
export PATH="$PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4

cd "$WORKING_DIR"
python3 compare_preconditioners.py
