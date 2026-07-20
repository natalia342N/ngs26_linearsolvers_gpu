from ngsolve import *
from netgen.occ import unit_square
import ngsolve.ngscuda as ngscuda
import time, os

maxh = 0.005   # ndof ~46665 — largest tested size

mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
fes  = H1(mesh, order=1, dirichlet=".*")
u, v = fes.TnT()
wind = CF((1, 0))
a    = BilinearForm((grad(u)*grad(v) + wind*grad(u)*v)*dx).Assemble()
f    = LinearForm(1*v*dx).Assemble()
pre  = a.mat.CreateSmoother(fes.FreeDofs())
pre1 = Projector(fes.FreeDofs(), True)

adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()
fdev    = f.vec.CreateDeviceVector(copy=True)
pa_dev  = predev @ adev
rhs_pre = (predev * fdev).Evaluate()

solver = ngscuda.DevTFQMRSolver(
    mat=pre @ a.mat, pre=pre1,
    adev_raw=pa_dev, cdev_raw=pre1dev,
    precision=1e-8, maxsteps=800, printrates=False)

gfu = GridFunction(fes)

t0 = time.perf_counter()
solver.Mult(rhs_pre, gfu.vec)
t1 = time.perf_counter()

use_graph = os.environ.get("NO_CUDA_GRAPH", "0") == "0"
print(f"ndof       = {fes.ndof}")
print(f"|sol|      = {Norm(gfu.vec):.8e}")
print(f"use_graph  = {use_graph}")
print(f"elapsed    = {(t1-t0)*1000:.3f} ms")
