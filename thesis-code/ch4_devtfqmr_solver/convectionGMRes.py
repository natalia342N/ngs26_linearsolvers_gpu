from ngsolve import *
from ngsolve.solvers import *
from ngsolve import la
from time import perf_counter
import scipy.sparse.linalg
import numpy as np
import ngsolve.ngscuda as ngscuda

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
TOL      = 1e-8
MAXSTEPS = 400
RUNS     = 5       # timed runs after 1 warm-up
WARMUPS  = 1

# -----------------------------------------------------------------------
# Problem setup — Problem 3: 3D DG convection (unit cube, L2 order 2)
# -----------------------------------------------------------------------
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.05))
fes  = L2(mesh, order=2, dgjumps=True)
print(f"ndof = {fes.ndof}")

u, v = fes.TnT()
wind = CF((1, 0.2, 0.3))
n    = specialcf.normal(mesh.dim)
dS   = dx(element_boundary=True)

a = BilinearForm(fes)
a += -20*u*v*dx
a += u*wind*grad(v)*dx
a += -(wind*n)*IfPos(wind*n, u, u.Other(bnd=0))*v*dS
with TaskManager():
    a.Assemble()

blocks = fes.CreateSmoothingBlocks(blocktype="element")
pre    = a.mat.CreateBlockSmoother(blocks)
pre1   = Projector(fes.FreeDofs(), True)
f      = LinearForm(1*v*dx).Assemble()
gfu    = GridFunction(fes)

print(f"tol={TOL}  maxsteps={MAXSTEPS}  runs={RUNS}  warmups={WARMUPS}")
print(f"On CPU, numthreads = {ngsglobals.numthreads}")

# -----------------------------------------------------------------------
# Reference solution (CPU TFQMR, tight tolerance)
# -----------------------------------------------------------------------
with TaskManager():
    gfu.vec.data = TFQMR(mat=pre@a.mat, pre=pre1,
                          rhs=(pre*f.vec).Evaluate(),
                          maxsteps=800, printrates=False, tol=1e-12)
ref_sol = gfu.vec.CreateVector()
ref_sol.data = gfu.vec
ref_norm = Norm(ref_sol)
print(f"Reference |sol| = {ref_norm:.10e}\n")

# -----------------------------------------------------------------------
# Timing helper
# -----------------------------------------------------------------------
results = []

def measure(name, fn, warmup_fn=None):
    """Run warmup_fn once, then fn RUNS times. Record mean, min, max time."""
    wfn = warmup_fn if warmup_fn is not None else fn
    iters_out = [0]

    # warm-up
    for _ in range(WARMUPS):
        iters_out[0] = wfn()

    times = []
    for _ in range(RUNS):
        t0 = perf_counter()
        iters_out[0] = fn()
        times.append(perf_counter() - t0)

    mean_t = np.mean(times)
    min_t  = np.min(times)
    max_t  = np.max(times)
    sol_norm = Norm(gfu.vec)
    err = abs(sol_norm - ref_norm)

    results.append((name, mean_t, min_t, max_t, iters_out[0], sol_norm, err))
    print(f"  {name:<40}  mean={mean_t*1e3:8.2f}ms  "
          f"[{min_t*1e3:.2f},{max_t*1e3:.2f}]  "
          f"iters={iters_out[0]:>4}  |sol|={sol_norm:.8e}  err={err:.1e}")
    return iters_out[0]

# -----------------------------------------------------------------------
# CPU solvers
# -----------------------------------------------------------------------
print("=== CPU solvers ===")

with TaskManager():
    def fn_gmres():
        gfu.vec.data = GMRes(A=a.mat, pre=pre, b=f.vec,
                             maxsteps=MAXSTEPS, printrates=False, tol=TOL)
        return MAXSTEPS  # NGSolve GMRes doesn't return iter count
    measure("CPU GMRes", fn_gmres)

with TaskManager():
    def fn_qmr():
        gfu.vec.data = QMR(mat=a.mat, pre1=pre1, pre2=pre, rhs=f.vec,
                           fdofs=fes.FreeDofs(), maxsteps=MAXSTEPS,
                           printrates=False)
        return MAXSTEPS
    measure("CPU QMR", fn_qmr)

with TaskManager():
    def fn_tfqmr():
        gfu.vec.data = TFQMR(mat=pre@a.mat, pre=pre1,
                              rhs=(pre*f.vec).Evaluate(),
                              maxsteps=MAXSTEPS, printrates=False, tol=TOL)
        return MAXSTEPS
    measure("CPU TFQMR", fn_tfqmr)

with TaskManager():
    def fn_cpp_gmres():
        gfu.vec.data = la.GMRESSolver(mat=a.mat, pre=pre,
                                      precision=TOL, maxsteps=MAXSTEPS,
                                      printrates=False) * f.vec
        return MAXSTEPS
    measure("CPU C++ GMRes", fn_cpp_gmres)

def fn_scipy_tfqmr():
    sol, info = scipy.sparse.linalg.tfqmr(
        A=pre@a.mat, b=(pre*f.vec).Evaluate(),
        M=pre1, rtol=TOL, maxiter=MAXSTEPS)
    gfu.vec.data = sol
    return info if isinstance(info, int) else MAXSTEPS
measure("CPU scipy TFQMR", fn_scipy_tfqmr)

# -----------------------------------------------------------------------
# GPU setup
# -----------------------------------------------------------------------
print("\n=== Moving operators to device ===")
fdev    = f.vec.CreateDeviceVector(copy=True)
adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()
pa_dev  = predev @ adev
rhs_pre = (predev * fdev).Evaluate()
print("Device operators ready.\n")

# -----------------------------------------------------------------------
# GPU solvers
# -----------------------------------------------------------------------
print("=== GPU solvers ===")

def fn_gpu_python_gmres():
    gfu.vec.data = GMRes(A=adev, pre=predev, b=fdev,
                         maxsteps=MAXSTEPS, printrates=False, tol=TOL)
    return MAXSTEPS
measure("GPU Python GMRes", fn_gpu_python_gmres)

def fn_gpu_qmr():
    gfu.vec.data = QMR(mat=adev, pre1=pre1dev, pre2=predev, rhs=fdev,
                       fdofs=fes.FreeDofs(), maxsteps=MAXSTEPS,
                       printrates=False, tol=TOL)
    return MAXSTEPS
measure("GPU QMR", fn_gpu_qmr)

udev = gfu.vec.CreateDeviceVector()
def fn_gpu_cpp_gmres():
    udev.data = la.GMRESSolver(mat=adev, pre=predev, precision=TOL,
                               maxsteps=MAXSTEPS, printrates=False) * fdev
    gfu.vec.data = udev
    return MAXSTEPS
measure("GPU C++ GMRes", fn_gpu_cpp_gmres)

def fn_gpu_scipy_tfqmr():
    sol, info = scipy.sparse.linalg.tfqmr(
        A=pa_dev, b=rhs_pre, M=None, rtol=TOL, maxiter=MAXSTEPS)
    gfu.vec.data = sol
    return info if isinstance(info, int) else MAXSTEPS
measure("GPU scipy TFQMR", fn_gpu_scipy_tfqmr)

def fn_gpu_python_tfqmr():
    gfu.vec.data = TFQMR(mat=pa_dev, pre=pre1dev, rhs=rhs_pre,
                          maxsteps=MAXSTEPS, printrates=False, tol=TOL)
    return MAXSTEPS
measure("GPU Python TFQMR", fn_gpu_python_tfqmr)

# DevTFQMR — warm-up triggers graph capture
solver = ngscuda.DevTFQMRSolver(mat=pre@a.mat, pre=pre1,
                                 adev_raw=pa_dev, cdev_raw=pre1dev,
                                 precision=TOL, maxsteps=MAXSTEPS,
                                 printrates=False)
def fn_devtfqmr_warmup():
    solver.Mult(rhs_pre, gfu.vec)
    return MAXSTEPS

def fn_devtfqmr():
    solver.Mult(rhs_pre, gfu.vec)
    return MAXSTEPS

measure("GPU DevTFQMR (WHILE graph)", fn_devtfqmr, warmup_fn=fn_devtfqmr_warmup)

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------
print("\n=== Summary table ===")
header = f"{'Solver':<40}  {'mean(ms)':>9}  {'min(ms)':>8}  {'max(ms)':>8}  {'iters':>6}  {'|sol|':>14}  {'err vs ref':>12}"
print(header)
print("-" * len(header))
for name, mean_t, min_t, max_t, iters, sol_norm, err in results:
    print(f"{name:<40}  {mean_t*1e3:9.2f}  {min_t*1e3:8.2f}  {max_t*1e3:8.2f}  "
          f"{str(iters):>6}  {sol_norm:14.8e}  {err:12.1e}")

# -----------------------------------------------------------------------
# CSV output
# -----------------------------------------------------------------------
print("\n=== CSV ===")
print("solver,mean_ms,min_ms,max_ms,iters,sol_norm,err_vs_ref")
for name, mean_t, min_t, max_t, iters, sol_norm, err in results:
    print(f"{name},{mean_t*1e3:.4f},{min_t*1e3:.4f},{max_t*1e3:.4f},"
          f"{iters},{sol_norm:.10e},{err:.4e}")
