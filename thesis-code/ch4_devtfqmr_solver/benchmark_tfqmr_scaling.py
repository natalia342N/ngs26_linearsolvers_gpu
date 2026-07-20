"""
Scaling benchmark: DevTFQMRSolver no-graph vs WHILE graph.
Problem: unit cube, DG L2 order 2, convection-diffusion (same as convectionGMRes.py).
Varying mesh size.
"""
import time
import os
from ngsolve import *
from ngsolve.solvers import *
from ngsolve import la
import ngsolve.ngscuda as ngscuda

RUNS = 5
tol  = 1e-12

print(f"CPU threads (NGSolve TaskManager): {ngsglobals.numthreads}")
print(f"{'ndof':>10}  {'Python TFQMR (ms)':>18}  {'No-graph (ms)':>14}  {'WHILE graph (ms)':>17}  {'Speedup':>8}")
print("-" * 77)

for maxh in [0.3, 0.2, 0.15, 0.1, 0.07, 0.05]:
    mesh = Mesh(unit_cube.GenerateMesh(maxh=maxh))
    fes  = L2(mesh, order=2, dgjumps=True)

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

    blocks  = fes.CreateSmoothingBlocks(blocktype="element")
    pre     = a.mat.CreateBlockSmoother(blocks)
    pre1    = Projector(fes.FreeDofs(), True)
    f       = LinearForm(1*v*dx).Assemble()
    gfu     = GridFunction(fes)

    adev    = a.mat.CreateDeviceMatrix()
    predev  = pre.CreateDeviceMatrix()
    pre1dev = pre1.CreateDeviceMatrix()
    fdev    = f.vec.CreateDeviceVector(copy=True)

    pa_dev  = predev @ adev
    rhs_pre = (predev * fdev).Evaluate()

    # --- Python TFQMR with device matrices ---
    from ngsolve.solvers import TFQMR as PyTFQMR
    gfu_tmp = GridFunction(fes)
    pa_dev  = predev @ adev
    rhs_pre = (predev * fdev).Evaluate()
    gfu_tmp.vec.data = PyTFQMR(mat=pa_dev, pre=pre1dev, rhs=rhs_pre,
                                maxsteps=1000, printrates=False, tol=tol)  # warmup
    t0 = time.perf_counter()
    for _ in range(RUNS):
        gfu_tmp.vec.data = PyTFQMR(mat=pa_dev, pre=pre1dev, rhs=rhs_pre,
                                    maxsteps=1000, printrates=False, tol=tol)
    py_ms = (time.perf_counter() - t0) / RUNS * 1000

    # --- DevTFQMR no-graph ---
    os.environ["NO_CUDA_GRAPH"] = "1"
    ng_solver = ngscuda.DevTFQMRSolver(mat=pre@a.mat, pre=pre1,
                                        adev_raw=pa_dev, cdev_raw=pre1dev,
                                        precision=tol, maxsteps=1000)
    ng_solver.Mult(rhs_pre, gfu.vec)  # warmup
    t0 = time.perf_counter()
    for _ in range(RUNS):
        ng_solver.Mult(rhs_pre, gfu.vec)
    ng_ms = (time.perf_counter() - t0) / RUNS * 1000
    del os.environ["NO_CUDA_GRAPH"]

    # --- DevTFQMR WHILE graph ---
    wh_solver = ngscuda.DevTFQMRSolver(mat=pre@a.mat, pre=pre1,
                                        adev_raw=pa_dev, cdev_raw=pre1dev,
                                        precision=tol, maxsteps=1000)
    wh_solver.Mult(rhs_pre, gfu.vec)  # warmup (builds graph)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        wh_solver.Mult(rhs_pre, gfu.vec)
    wh_ms = (time.perf_counter() - t0) / RUNS * 1000

    # correctness: compare GPU solution norm to Python TFQMR reference
    ref_norm = Norm(gfu_tmp.vec)
    gpu_norm = Norm(gfu.vec)
    rel_err  = abs(gpu_norm - ref_norm) / (ref_norm + 1e-30)

    ok = "OK" if rel_err < 1e-6 else "FAIL"
    print(f"{fes.ndof:>10}  {py_ms:>18.1f}  {ng_ms:>14.1f}  {wh_ms:>17.1f}  {ng_ms/wh_ms:>7.2f}×  "
          f"err={rel_err:.1e} {ok}")
