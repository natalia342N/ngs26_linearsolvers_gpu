#!/bin/bash
#SBATCH --job-name=convection
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ml --force purge
ml load ASC/2023.06
ml load buildenv/default-foss-2023a
ml load CMake/3.26.3-GCCcore-12.3.0 CUDA/12.9.0 SciPy-bundle/2023.07-gfbf-2023a occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

source ~/starting_with_ngscuda/ngs/bin/activate
export PYTHONPATH=$(find ~/starting_with_ngscuda/install -maxdepth 6 -type d -name site-packages | head -1)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
srun --ntasks=1 python ~/starting_with_ngscuda/convectionGMRes.py
