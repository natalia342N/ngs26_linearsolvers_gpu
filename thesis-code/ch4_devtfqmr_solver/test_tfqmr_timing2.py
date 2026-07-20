import ngsolve
print(f"NGSolve: {ngsolve.__version__}")
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from ngsolve import la
from netgen.occ import unit_cube
from time import perf_counter
import os

mesh = Mesh(unit_cube.GenerateMesh(maxh=0.05))
fes  = L2(mesh, order=2, dgjumps=True)
u, v = fes.TnT()
wind = CF((1, 0.2, 0.3))
n    = specialcf.normal(mesh.dim)
dS   = dx(element_boundary=True)

a = BilinearForm(fes)
a += -20*u*v*dx
a += u*wind*grad(v)*dx
a += -(wind*n)*IfPos(wind*n, u, u.Other(bnd=0))*v*dS
a.Assemble()

blocks = fes.CreateSmoothingBlocks(blocktype="element")
pre    = a.mat.CreateBlockSmoother(blocks)
pre1   = Projector(fes.FreeDofs(), True)
f      = LinearForm(1*v*dx).Assemble()
gfu    = GridFunction(fes)

fdev    = f.vec.CreateDeviceVector(copy=True)
adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()

print(f"ndof = {fes.ndof}")

ITERS = 105  # known from verbose output

# ── use_graph=1: first call (includes warm-up+capture) ──
t0 = perf_counter()
solver = ngscuda.DevTFQMRSolver(
    mat=predev@adev, pre=pre1dev,
    adev_raw=adev, cdev_raw=predev, ddev_raw=pre1dev,
    precision=1e-12, maxsteps=400, printrates=False)
gfu.vec.data = solver * (predev*fdev).Evaluate()
t1 = perf_counter()
time_graph_first = t1 - t0
sol_graph = Norm(gfu.vec)

# subsequent calls (graph already built — just replay)
N = 5
t0 = perf_counter()
for _ in range(N):
    gfu.vec.data = solver * (predev*fdev).Evaluate()
t1 = perf_counter()
time_graph_subsequent = (t1-t0)/N

# ── use_graph=0 ──────────────────────────────────────────
os.environ['NO_CUDA_GRAPH'] = '1'
solver_ng = ngscuda.DevTFQMRSolver(
    mat=predev@adev, pre=pre1dev,
    adev_raw=adev, cdev_raw=predev, ddev_raw=pre1dev,
    precision=1e-12, maxsteps=400, printrates=False)

t0 = perf_counter()
for _ in range(N):
    gfu.vec.data = solver_ng * (predev*fdev).Evaluate()
t1 = perf_counter()
time_nograph = (t1-t0)/N
sol_nograph = Norm(gfu.vec)
del os.environ['NO_CUDA_GRAPH']

# ── Python TFQMR reference ───────────────────────────────
from ngsolve.solvers import TFQMR
rhs_dev = (predev*fdev).Evaluate()
t0 = perf_counter()
gfu.vec.data = TFQMR(mat=predev@adev, pre=pre1dev,
                     rhs=rhs_dev, maxsteps=400,
                     printrates=False, tol=1e-12)
t1 = perf_counter()
time_python = t1 - t0
sol_python = Norm(gfu.vec)

# ── CPU reference ─────────────────────────────────────────
from ngsolve import la
t0 = perf_counter()
gfu.vec.data = la.GMRESSolver(adev, predev,
                               precision=1e-12,
                               maxsteps=400,
                               printrates=False) * fdev
t1 = perf_counter()
time_gmres = t1 - t0

print(f"\n{'='*60}")
print(f"TFQMR Timing Breakdown  (ndof={fes.ndof}, iters={ITERS})")
print(f"{'='*60}")
print(f"\n── Total solve time ──────────────────────────────────")
print(f"  DevTFQMR graph=1  (1st call, incl warm-up+capture): "
      f"{time_graph_first*1000:.2f}ms")
print(f"  DevTFQMR graph=1  (subsequent, graph reused):        "
      f"{time_graph_subsequent*1000:.2f}ms")
print(f"  DevTFQMR graph=0  (no graph, avg {N} runs):          "
      f"{time_nograph*1000:.2f}ms")
print(f"  Python TFQMR GPU  (baseline):                        "
      f"{time_python*1000:.2f}ms")
print(f"  C++ GMRESSolver GPU:                                  "
      f"{time_gmres*1000:.2f}ms")

print(f"\n── Per-iteration time ────────────────────────────────")
print(f"  DevTFQMR graph=1  (subsequent): "
      f"{time_graph_subsequent*1000/ITERS:.4f}ms/iter")
print(f"  DevTFQMR graph=0:               "
      f"{time_nograph*1000/ITERS:.4f}ms/iter")

print(f"\n── Graph benefit analysis ────────────────────────────")
warmup_cost = (time_graph_first - time_graph_subsequent)*1000
graph_saving = (time_nograph - time_graph_subsequent)*1000
print(f"  Warm-up cost (1st call overhead): {warmup_cost:.2f}ms")
print(f"  Graph saving per solve:           {graph_saving:.2f}ms")
print(f"  Graph saving per iteration:       "
      f"{graph_saving/ITERS*1000:.2f}µs")
print(f"  Graph speedup (subsequent calls): "
      f"{graph_saving/(time_nograph*1000)*100:.2f}%")

print(f"\n── Launch overhead (from nsys) ───────────────────────")
print(f"  cudaGraphLaunch per iter:   ~6.3µs × 2 = ~12.6µs")
print(f"  cudaLaunchKernel × 6/iter:  ~2µs × 6  = ~12µs")
print(f"  Net saved per iter:         ~0µs (similar)")
print(f"  Note: dominant cost is DtoH sync ~140ms/iter")
print(f"  Graph benefit fraction of total: "
      f"~{12.6*ITERS/(140000*ITERS)*100:.3f}%")

print(f"\n── Solution verification ─────────────────────────────")
print(f"  graph=1: |sol| = {sol_graph:.8e}")
print(f"  graph=0: |sol| = {sol_nograph:.8e}")
print(f"  python:  |sol| = {sol_python:.8e}")
print(f"  match: {abs(sol_graph - sol_nograph) < 1e-6}")
print(f"{'='*60}")
