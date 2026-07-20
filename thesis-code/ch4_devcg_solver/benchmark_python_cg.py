"""
Benchmark: Python CGSolver (tutorial 5.5.1) vs DevCGSolver no-graph vs DevCGSolver WHILE graph.
Problem: unit square, H1 order 2, Poisson, BlockSmoother preconditioner.
Same setup as test_devcg_scaling.py.
"""
import time
from ngsolve import *
from ngsolve.krylovspace import CGSolver
from netgen.geom2d import unit_square
import ngsolve.ngscuda as ngscuda
import os

RUNS = 3

sizes = [0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03]

print(f"{'ndof':>10}  {'Python CG (ms)':>15}  {'DevCG no-graph (ms)':>20}  {'DevCG WHILE (ms)':>18}  {'speedup (WHILE/Python)':>22}")
print("-" * 95)

for maxh in sizes:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    f = LinearForm(x*y*v*dx).Assemble()
    gfu = GridFunction(fes)

    blocks = fes.CreateSmoothingBlocks()
    pre    = a.mat.CreateBlockSmoother(blocks)
    adev   = a.mat.CreateDeviceMatrix()
    predev = pre.CreateDeviceMatrix()
    fdev   = f.vec.CreateDeviceVector(copy=True)

    # --- Python CGSolver (tutorial 5.5.1) ---
    py_solver = CGSolver(adev, predev, maxiter=1000, printrates=False, tol=1e-10)
    gfu.vec.data = (py_solver * fdev).Evaluate()  # warmup
    t0 = time.perf_counter()
    for _ in range(RUNS):
        gfu.vec.data = (py_solver * fdev).Evaluate()
    py_ms = (time.perf_counter() - t0) / RUNS * 1000

    # --- DevCGSolver no-graph ---
    os.environ["NO_CUDA_GRAPH"] = "1"
    ng_solver = ngscuda.DevCGSolver(adev, predev, adev_raw=adev, cdev_raw=predev,
                                     precision=1e-10, maxsteps=1000, printrates=False)
    ng_solver.Mult(fdev, gfu.vec)  # warmup
    t0 = time.perf_counter()
    for _ in range(RUNS):
        ng_solver.Mult(fdev, gfu.vec)
    ng_ms = (time.perf_counter() - t0) / RUNS * 1000
    del os.environ["NO_CUDA_GRAPH"]

    # --- DevCGSolver WHILE graph ---
    wh_solver = ngscuda.DevCGSolver(adev, predev, adev_raw=adev, cdev_raw=predev,
                                     precision=1e-10, maxsteps=1000, printrates=False)
    wh_solver.Mult(fdev, gfu.vec)  # warmup (builds graph)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        wh_solver.Mult(fdev, gfu.vec)
    wh_ms = (time.perf_counter() - t0) / RUNS * 1000

    speedup = py_ms / wh_ms
    print(f"{fes.ndof:>10}  {py_ms:>15.3f}  {ng_ms:>20.3f}  {wh_ms:>18.3f}  {speedup:>22.2f}x")
