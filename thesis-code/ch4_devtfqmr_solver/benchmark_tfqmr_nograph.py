"""
Benchmark: Python TFQMR vs DevTFQMRSolver no-graph vs DevTFQMRSolver WHILE graph.
Problem: unit cube, DG L2 order 2, convection-diffusion (same as convectionGMRes.py).
"""
import time
import os
from ngsolve import *
from ngsolve.solvers import *
from ngsolve import la
import ngsolve.ngscuda as ngscuda

RUNS = 5
tol  = 1e-8

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

adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()
fdev    = f.vec.CreateDeviceVector(copy=True)

pa_dev  = predev @ adev
rhs_pre = (predev * fdev).Evaluate()

# --- Python TFQMR ---
gfu.vec.data = TFQMR(mat=pa_dev, pre=pre1dev, rhs=rhs_pre, maxsteps=400, printrates=False, tol=tol)  # warmup
t0 = time.perf_counter()
for _ in range(RUNS):
    gfu.vec.data = TFQMR(mat=pa_dev, pre=pre1dev, rhs=rhs_pre, maxsteps=400, printrates=False, tol=tol)
py_ms = (time.perf_counter() - t0) / RUNS * 1000

# --- DevTFQMR no-graph ---
os.environ["NO_CUDA_GRAPH"] = "1"
ng_solver = ngscuda.DevTFQMRSolver(mat=pre@a.mat, pre=pre1,
                                    adev_raw=pa_dev, cdev_raw=pre1dev,
                                    precision=tol, maxsteps=400)
ng_solver.Mult(rhs_pre, gfu.vec)  # warmup
t0 = time.perf_counter()
for _ in range(RUNS):
    ng_solver.Mult(rhs_pre, gfu.vec)
ng_ms = (time.perf_counter() - t0) / RUNS * 1000
del os.environ["NO_CUDA_GRAPH"]

# --- DevTFQMR WHILE graph ---
wh_solver = ngscuda.DevTFQMRSolver(mat=pre@a.mat, pre=pre1,
                                    adev_raw=pa_dev, cdev_raw=pre1dev,
                                    precision=tol, maxsteps=400)
wh_solver.Mult(rhs_pre, gfu.vec)  # warmup (builds graph)
t0 = time.perf_counter()
for _ in range(RUNS):
    wh_solver.Mult(rhs_pre, gfu.vec)
wh_ms = (time.perf_counter() - t0) / RUNS * 1000

print(f"\n{'Solver':<35} {'Time (ms)':>10}  {'vs Python':>10}  {'vs No-graph':>12}")
print("-" * 72)
print(f"{'Python TFQMR':<35} {py_ms:>10.1f}  {'1×':>10}  {'—':>12}")
print(f"{'DevTFQMR no-graph':<35} {ng_ms:>10.1f}  {py_ms/ng_ms:>9.2f}×  {'1×':>12}")
print(f"{'DevTFQMR WHILE graph':<35} {wh_ms:>10.1f}  {py_ms/wh_ms:>9.2f}×  {ng_ms/wh_ms:>11.2f}×")
