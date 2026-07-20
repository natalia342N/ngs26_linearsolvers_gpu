"""
CG scaling benchmark with Jacobi smoother (CreateSmoother) — matches tutorial 5.5.1.
Four-way: CPU CGSolver vs C++ CGSolver (device) vs DevCGSolver no-graph vs WHILE graph.
Uses ngsolve.la.CGSolver (C++ version) to match the 5.5.1 tutorial baseline.
Problem: unit square, H1 order 2, Poisson.
"""
import time
import os
import ngsolve
from ngsolve import *
from netgen.geom2d import unit_square
import ngsolve.ngscuda as ngscuda

RUNS = 3

default_threads = ngsglobals.numthreads
print(f"CPU threads (NGSolve TaskManager): {default_threads}")
print(f"NGSolve version: {ngsolve.__version__}")

sizes = [0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003]

print(f"{'ndof':>10}  {'CPU all-T (ms)':>15}  {'CPU 1T (ms)':>12}  {'C++ dev (ms)':>13}  {'No-graph (ms)':>14}  {'WHILE graph (ms)':>17}  {'vs 1T':>6}  {'vs C++ dev':>11}  {'vs No-graph':>12}")
print("-" * 140)

for maxh in sizes:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    with TaskManager():
        a = BilinearForm(grad(u)*grad(v)*dx + u*v*dx).Assemble()
        f = LinearForm(x*v*dx).Assemble()
    gfu = GridFunction(fes)

    jac  = a.mat.CreateSmoother(fes.FreeDofs())
    adev = a.mat.CreateDeviceMatrix()
    jdev = jac.CreateDeviceMatrix()
    fdev = f.vec.CreateDeviceVector(copy=True)

    # --- CPU CGSolver (all available threads) ---
    with TaskManager():
        cpu_solver = CGSolver(a.mat, jac, maxsteps=2000, precision=1e-12, printrates=False)
        gfu.vec.data = cpu_solver * f.vec  # warmup
        t0 = time.perf_counter()
        for _ in range(RUNS):
            gfu.vec.data = cpu_solver * f.vec
        cpu_ms = (time.perf_counter() - t0) / RUNS * 1000

    # --- CPU CGSolver single-threaded (no TaskManager) ---
    cpu_solver1 = CGSolver(a.mat, jac, maxsteps=2000, precision=1e-12, printrates=False)
    gfu.vec.data = cpu_solver1 * f.vec  # warmup
    t0 = time.perf_counter()
    for _ in range(RUNS):
        gfu.vec.data = cpu_solver1 * f.vec
    cpu_1t_ms = (time.perf_counter() - t0) / RUNS * 1000

    # --- C++ CGSolver with device matrices ---
    dev_solver = CGSolver(adev, jdev, maxsteps=2000, precision=1e-12, printrates=False)
    gfu.vec.data = (dev_solver * fdev).Evaluate()  # warmup
    t0 = time.perf_counter()
    for _ in range(RUNS):
        gfu.vec.data = (dev_solver * fdev).Evaluate()
    py_dev_ms = (time.perf_counter() - t0) / RUNS * 1000

    # --- DevCGSolver no-graph ---
    os.environ["NO_CUDA_GRAPH"] = "1"
    ng_solver = ngscuda.DevCGSolver(adev, jdev, adev_raw=adev, cdev_raw=jdev,
                                     precision=1e-12, maxsteps=2000, printrates=False)
    ng_solver.Mult(fdev, gfu.vec)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        ng_solver.Mult(fdev, gfu.vec)
    ng_ms = (time.perf_counter() - t0) / RUNS * 1000
    del os.environ["NO_CUDA_GRAPH"]

    # --- DevCGSolver WHILE graph ---
    wh_solver = ngscuda.DevCGSolver(adev, jdev, adev_raw=adev, cdev_raw=jdev,
                                     precision=1e-12, maxsteps=2000, printrates=False)
    wh_solver.Mult(fdev, gfu.vec)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        wh_solver.Mult(fdev, gfu.vec)
    wh_ms = (time.perf_counter() - t0) / RUNS * 1000

    print(f"{fes.ndof:>10}  {cpu_ms:>15.1f}  {cpu_1t_ms:>12.1f}  {py_dev_ms:>13.1f}  {ng_ms:>14.1f}  {wh_ms:>17.1f}  {cpu_1t_ms/wh_ms:>5.1f}×  {py_dev_ms/wh_ms:>10.2f}×  {ng_ms/wh_ms:>11.2f}×")

print()
print(f"GPU:     NVIDIA H100 (single card)")
print(f"CPU:     AMD EPYC Zen4, dual-socket, {default_threads} logical threads available on node")
print(f"         CPU all-T: NGSolve TaskManager used all {default_threads} threads (ignores SLURM --cpus-per-task=22)")
print(f"         CPU 1T:    no TaskManager, single-threaded reference")
print(f"Software: NGSolve {ngsolve.__version__}, CUDA 12.9")
