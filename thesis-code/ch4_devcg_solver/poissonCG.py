from ngsolve import *
from ngsolve.solvers import *
from ngsolve.krylovspace import MinResSolver, MinRes
from ngsolve import la
from time import perf_counter
import scipy.sparse.linalg
import numpy as np
import ngsolve.ngscuda as ngscuda

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
TOL      = 1e-8
MAXSTEPS = 1000
RUNS     = 5       # timed runs after warm-up
WARMUPS  = 1

# -----------------------------------------------------------------------
# Problem setup — 3D Poisson (unit cube, H1 order 2, Dirichlet BC)
# Symmetric positive definite → suitable for CG and MinRes
# -----------------------------------------------------------------------
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.07))
fes  = H1(mesh, order=2, dirichlet=".*")
print(f"ndof = {fes.ndof}")

u, v = fes.TnT()
a = BilinearForm(fes)
a += grad(u) * grad(v) * dx
with TaskManager():
    a.Assemble()

f = LinearForm(x * y * v * dx).Assemble()

blocks = fes.CreateSmoothingBlocks()
pre    = a.mat.CreateBlockSmoother(blocks)
pre1   = Projector(fes.FreeDofs(), True)
gfu    = GridFunction(fes)

print(f"tol={TOL}  maxsteps={MAXSTEPS}  runs={RUNS}  warmups={WARMUPS}")
print(f"On CPU, numthreads = {ngsglobals.numthreads}")

# -----------------------------------------------------------------------
# Reference solution (CPU CG, tight tolerance)
# -----------------------------------------------------------------------
_cg_ref = la.CGSolver(mat=a.mat, pre=pre,
                       precision=1e-14, maxsteps=5000, printrates=False)
with TaskManager():
    gfu.vec.data = _cg_ref * f.vec
ref_sol = gfu.vec.CreateVector()
ref_sol.data = gfu.vec
ref_norm = Norm(ref_sol)
print(f"Reference |sol| = {ref_norm:.10e}  (iters={_cg_ref.GetSteps()})\n")

# -----------------------------------------------------------------------
# Timing helper
# -----------------------------------------------------------------------
results = []

def measure(name, fn, warmup_fn=None):
    """Run warmup once, then fn RUNS times. Record mean/min/max time."""
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
    print(f"  {name:<44}  mean={mean_t*1e3:8.2f}ms  "
          f"[{min_t*1e3:.2f},{max_t*1e3:.2f}]  "
          f"iters={iters_out[0]:>6}  |sol|={sol_norm:.8e}  err={err:.1e}")
    return iters_out[0]

# -----------------------------------------------------------------------
# CPU solvers
# -----------------------------------------------------------------------
print("=== CPU solvers ===")

with TaskManager():
    def fn_cpu_py_cg():
        solver = CGSolver(mat=a.mat, pre=pre, tol=TOL, maxiter=MAXSTEPS,
                          printrates=False)
        solver.Solve(rhs=f.vec, sol=gfu.vec)
        return solver.iterations
    measure("CPU Python CG", fn_cpu_py_cg)

with TaskManager():
    def fn_cpu_py_minres():
        solver = MinResSolver(mat=a.mat, pre=pre, tol=TOL, maxiter=MAXSTEPS,
                              printrates=False)
        solver.Solve(rhs=f.vec, sol=gfu.vec)
        return solver.iterations
    measure("CPU Python MinRes", fn_cpu_py_minres)

with TaskManager():
    def fn_cpu_cpp_cg():
        s = la.CGSolver(mat=a.mat, pre=pre,
                        precision=TOL, maxsteps=MAXSTEPS, printrates=False)
        gfu.vec.data = s * f.vec
        return s.GetSteps()
    measure("CPU C++ CGSolver", fn_cpu_cpp_cg)

def fn_scipy_cg():
    iters = [0]
    sol, info = scipy.sparse.linalg.cg(
        A=a.mat, b=f.vec, M=pre,
        rtol=TOL, maxiter=MAXSTEPS,
        callback=lambda xk: iters.__setitem__(0, iters[0] + 1))
    gfu.vec.data = sol
    return iters[0]
measure("CPU scipy CG", fn_scipy_cg)

def fn_scipy_minres():
    iters = [0]
    sol, info = scipy.sparse.linalg.minres(
        A=a.mat, b=f.vec, M=pre,
        rtol=TOL, maxiter=MAXSTEPS,
        callback=lambda xk: iters.__setitem__(0, iters[0] + 1))
    gfu.vec.data = sol
    return iters[0]
measure("CPU scipy MinRes", fn_scipy_minres)

# -----------------------------------------------------------------------
# GPU setup
# -----------------------------------------------------------------------
print("\n=== Moving operators to device ===")
fdev   = f.vec.CreateDeviceVector(copy=True)
adev   = a.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()
print("Device operators ready.\n")

# -----------------------------------------------------------------------
# GPU solvers
# -----------------------------------------------------------------------
print("=== GPU solvers ===")

def fn_gpu_py_cg():
    solver = CGSolver(mat=adev, pre=predev, tol=TOL, maxiter=MAXSTEPS,
                      printrates=False)
    solver.Solve(rhs=fdev, sol=gfu.vec)
    return solver.iterations
measure("GPU Python CG", fn_gpu_py_cg)

udev = gfu.vec.CreateDeviceVector()

def fn_gpu_cpp_cg():
    s = la.CGSolver(mat=adev, pre=predev,
                    precision=TOL, maxsteps=MAXSTEPS, printrates=False)
    udev.data = s * fdev
    gfu.vec.data = udev
    return s.GetSteps()
measure("GPU C++ CGSolver (no graph)", fn_gpu_cpp_cg)

# DevCGSolver — warm-up triggers WHILE graph capture
cg_solver = ngscuda.DevCGSolver(mat=a.mat, pre=pre,
                                  adev_raw=adev, cdev_raw=predev,
                                  precision=TOL, maxsteps=MAXSTEPS,
                                  printrates=False)

def fn_devcg_warmup():
    cg_solver.Mult(fdev, udev)
    gfu.vec.data = udev
    return cg_solver.GetSteps()

def fn_devcg():
    cg_solver.Mult(fdev, udev)
    gfu.vec.data = udev
    return cg_solver.GetSteps()

measure("GPU DevCGSolver (WHILE graph)", fn_devcg, warmup_fn=fn_devcg_warmup)

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------
print("\n=== Summary table ===")
header = (f"{'Solver':<44}  {'mean(ms)':>9}  {'min(ms)':>8}  {'max(ms)':>8}  "
          f"{'iters':>6}  {'|sol|':>14}  {'err vs ref':>12}")
print(header)
print("-" * len(header))
for name, mean_t, min_t, max_t, iters, sol_norm, err in results:
    print(f"{name:<44}  {mean_t*1e3:9.2f}  {min_t*1e3:8.2f}  {max_t*1e3:8.2f}  "
          f"{str(iters):>6}  {sol_norm:14.8e}  {err:12.1e}")

# -----------------------------------------------------------------------
# CSV output
# -----------------------------------------------------------------------
print("\n=== CSV ===")
print("solver,mean_ms,min_ms,max_ms,iters,sol_norm,err_vs_ref")
for name, mean_t, min_t, max_t, iters, sol_norm, err in results:
    print(f"{name},{mean_t*1e3:.4f},{min_t*1e3:.4f},{max_t*1e3:.4f},"
          f"{iters},{sol_norm:.10e},{err:.4e}")
