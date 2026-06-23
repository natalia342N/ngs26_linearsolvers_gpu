# Musica (EESSI / ASC 2025)

This page is optional and covers building NGSolve with CUDA support and running GPU jobs
on the Musica supercomputer at TU Wien using the 2025 software stack.

---

## 1. Log in

```bash
ssh username@musica.vie.asc.ac.at
```

If you get a host key warning after a system update:
```bash
ssh-keygen -R musica.vie.asc.ac.at
```

---

## 2. Build NGSolve with CUDA

There are three build levels depending on your situation:

| Level | Script | When to use | Time |
|---|---|---|---|
| **Full** (`USE_SUPERBUILD=ON`) | see note below | new install, Netgen not yet built | 
| **NGSolve-only** (`USE_SUPERBUILD=OFF`) | `build_ngsolve.sh` | Netgen already installed, rebuilding NGSolve | 
| **ngscuda-only** | `make -j8` in build dir | changed only `ngscuda/` source files |

The script below is **NGSolve-only** — the most common case when iterating on ngscuda.
It assumes Netgen is already installed at `${WORKING_DIR}/install`.
For a full from-scratch build, add `-DUSE_SUPERBUILD=ON` and remove `-DNETGEN_DIR`.

---

Create a working directory and clone NGSolve:

```bash
mkdir ~/ngscuda && cd ~/ngscuda
git clone --recurse-submodules https://github.com/NGSolve/ngsolve.git src/ngsolve
```

Save the following as `build_ngsolve.sh` and run it **on the login node**
(GPU is optional for the build):

```bash
#!/bin/sh
set -e

ml --force purge
ml load ASC/2025.06
ml load CMake/3.31.3-GCCcore-14.2.0
ml load CUDA/12.9.1
ml load SciPy-bundle/2024.05-gfbf-2024a
ml load occt/7.9.1-GCCcore-14.2.0
ml unload pybind11 || true

CUDA_LIB=/cvmfs/software.asc.ac.at/versions/2025.06/software/linux/x86_64/amd/zen4/software/CUDA/12.9.1/targets/x86_64-linux/lib
export LIBRARY_PATH=$CUDA_LIB:$CUDA_LIB/stubs:$LIBRARY_PATH

WORKING_DIR=$(realpath "${PWD}")
VENV="${WORKING_DIR}/ngs"
SOURCES="${WORKING_DIR}/src/ngsolve"

virtualenv "${VENV}"
source "${VENV}/bin/activate"

BUILD_DIR="${WORKING_DIR}/build/ngsolve"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

GCC14LIB=$(find /cvmfs/software.eessi.io -path "*/GCCcore/14.2.0/lib64/libstdc++.so.6" 2>/dev/null | head -1)
GCC14DIR=$(dirname "$GCC14LIB")

cmake "${SOURCES}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SUPERBUILD=OFF \
  -DNETGEN_DIR="${WORKING_DIR}/install/lib/cmake/netgen" \
  -DUSE_OCC=ON \
  -DUSE_CCACHE=ON \
  -DCMAKE_INSTALL_PREFIX="${WORKING_DIR}/install" \
  -DUSE_CUDA=ON \
  -DUSE_GUI=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="90" \
  -DUSE_UMFPACK=OFF \
  -DBUILD_STUB_FILES=OFF

make -j8 kernel_generator
patchelf --set-rpath "$GCC14DIR" basiclinalg/kernel_generator 2>/dev/null || true

make -j8 install
pip install numpy scipy -q

echo "=== Build done ==="
echo "Install prefix: ${WORKING_DIR}/install"
```

Run with:
```bash
bash build_ngsolve.sh
```

### Fast rebuild (ngscuda changes only)

After making changes to `ngscuda/` source files, you don't need a full rebuild.
Run this from the login node:

```bash
ml --force purge
ml load ASC/2025.06
ml load CMake/3.31.3-GCCcore-14.2.0 CUDA/12.9.1
ml load SciPy-bundle/2024.05-gfbf-2024a occt/7.9.1-GCCcore-14.2.0
ml unload pybind11 || true

cd ~/ngscuda/build/ngsolve/ngscuda
make -j8 && make install
```

---

## 3. Submit a GPU job

Save the following as `submit.sh`, adjusting `WORKING_DIR` and your Python script path:

```bash
#!/bin/bash
#SBATCH --job-name=ngscuda_job
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ml --force purge
ml load ASC/2025.06
ml load CUDA/12.9.0
ml load SciPy-bundle/2025.06-gfbf-2025a
ml load occt/7.9.1-GCCcore-14.2.0

WORKING_DIR="/home/$USER/ngscuda"
PREFIX="$WORKING_DIR/install"

GCC14_LIB=/cvmfs/software.eessi.io/versions/2025.06/software/linux/x86_64/amd/zen4/software/GCCcore/14.2.0/lib64
OCCT_LIB=/cvmfs/software.eessi.io/versions/2025.06/software/linux/x86_64/amd/zen4/software/occt/7.9.1-GCCcore-14.2.0/lib
CUDA_LIB="${CUDA_HOME}/lib64"
export LD_LIBRARY_PATH=${GCC14_LIB}:${OCCT_LIB}:${CUDA_LIB}:${LD_LIBRARY_PATH:-}

PYDIR="$PREFIX/lib/python3.13/site-packages"
export PYTHONPATH="$PYDIR"
export PATH="$PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== NGSolve ==="
python3 -c "import ngsolve; print('version:', ngsolve.__version__)"
python3 -c "import ngsolve.ngscuda; print('ngscuda: OK')"

echo "=== Running ==="
python3 my_script.py
```

Submit with:
```bash
sbatch submit.sh
```

Check output in `ngscuda_job-<jobid>.out` and `.err`.

---

## 4. Run a Jupyter notebook on GPU

To execute a notebook and save outputs:

```bash
#!/bin/bash
#SBATCH --job-name=run_notebook
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

ml --force purge
ml load ASC/2025.06
ml load CUDA/12.9.0
ml load SciPy-bundle/2025.06-gfbf-2025a
ml load occt/7.9.1-GCCcore-14.2.0
ml load JupyterNotebook

WORKING_DIR="/home/$USER/ngscuda"
PREFIX="$WORKING_DIR/install"

GCC14_LIB=/cvmfs/software.eessi.io/versions/2025.06/software/linux/x86_64/amd/zen4/software/GCCcore/14.2.0/lib64
OCCT_LIB=/cvmfs/software.eessi.io/versions/2025.06/software/linux/x86_64/amd/zen4/software/occt/7.9.1-GCCcore-14.2.0/lib
CUDA_LIB="${CUDA_HOME}/lib64"
export LD_LIBRARY_PATH=${GCC14_LIB}:${OCCT_LIB}:${CUDA_LIB}:${LD_LIBRARY_PATH:-}

PYDIR="$PREFIX/lib/python3.13/site-packages"
NUMPY_DIR=$(python3 -c "import numpy, os; print(os.path.dirname(os.path.dirname(numpy.__file__)))")
export PYTHONPATH="$PYDIR:$NUMPY_DIR"
export PATH="$PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1

cd "$WORKING_DIR"
jupyter nbconvert \
    --to notebook \
    --execute \
    --allow-errors \
    --ExecutePreprocessor.timeout=600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output my_notebook_executed.ipynb \
    my_notebook.ipynb
```

---

## 5. Quick test

To verify the build works before moving on:

```python
# smoke_test.py
import ngsolve
from ngsolve import *
from ngsolve.ngscuda import *

print("NGSolve version:", ngsolve.__version__)

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()
a    = BilinearForm(grad(u)*grad(v)*dx + u*v*dx).Assemble()
f    = LinearForm(1*v*dx).Assemble()
jac  = a.mat.CreateSmoother(fes.FreeDofs())

adev   = a.mat.CreateDeviceMatrix()
jacdev = jac.CreateDeviceMatrix()
fdev   = f.vec.CreateDeviceVector(copy=True)

solver = DevCGSolver(mat=adev, pre=jacdev, adev_raw=adev, cdev_raw=jacdev,
                     maxsteps=500, printrates=False)
gfu = GridFunction(fes)
gfu.vec.data = (solver * fdev).Evaluate()
print("DevCGSolver OK, |sol| =", Norm(gfu.vec))
```

---

## Module versions (tested, June 2026)

| Component | Module |
|---|---|
| Software stack | `ASC/2025.06` |
| CUDA (build) | `CUDA/12.9.1` |
| CUDA (run) | `CUDA/12.9.0` |
| Python | 3.13 |
| SciPy | `SciPy-bundle/2025.06-gfbf-2025a` |
| OCC | `occt/7.9.1-GCCcore-14.2.0` |
| GPU | NVIDIA H100 (sm_90) |
