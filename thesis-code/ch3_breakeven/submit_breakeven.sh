#!/bin/bash
#SBATCH --job-name=breakeven
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ml --force purge
ml load ASC/2025.06
ml load CUDA/12.9.0
ml load SciPy-bundle/2025.06-gfbf-2025a
ml load occt/7.9.1-GCCcore-14.2.0

WORKING_DIR="/home/nt51825/starting_with_ngscuda"

GCC14_LIB=/cvmfs/software.eessi.io/versions/2025.06/software/linux/x86_64/amd/zen4/software/GCCcore/14.2.0/lib64
CUDA_LIB="${CUDA_HOME}/lib64"
export LD_LIBRARY_PATH=${GCC14_LIB}:${CUDA_LIB}:${LD_LIBRARY_PATH:-}

cd "$WORKING_DIR"

# install matplotlib if missing
python3 -c "import matplotlib" 2>/dev/null || pip install --user matplotlib

echo "=== Running benchmark ==="
./benchmark_breakeven_bin | tee breakeven_data.csv
echo "=== Plotting ==="
python3 plot_breakeven.py breakeven_data.csv
echo "=== Done ==="
