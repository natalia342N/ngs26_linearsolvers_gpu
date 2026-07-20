#!/bin/bash
#SBATCH --job-name=diamond_heavy
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ml --force purge
ml load ASC/2023.06
ml load CUDA/12.9.0

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Compile — override ITERS/N/NBLOCKS here if you want to tune without editing the .cu
nvcc -O2 -arch=sm_90 \
    -DN=1048576 \
    -DITERS=200 \
    -DNBLOCKS=64 \
    test_diamond_heavy.cu -o test_diamond_heavy

echo "=== ITERS=200 (baseline) ==="
./test_diamond_heavy

echo ""
echo "=== ITERS=1000 (heavier branches) ==="
nvcc -O2 -arch=sm_90 -DN=1048576 -DITERS=1000 -DNBLOCKS=64 \
    test_diamond_heavy.cu -o test_diamond_heavy_1000
./test_diamond_heavy_1000

echo ""
echo "=== ITERS=5000 (heavy branches) ==="
nvcc -O2 -arch=sm_90 -DN=1048576 -DITERS=5000 -DNBLOCKS=64 \
    test_diamond_heavy.cu -o test_diamond_heavy_5000
./test_diamond_heavy_5000
