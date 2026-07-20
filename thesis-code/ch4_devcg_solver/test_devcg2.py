import ngsolve
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from netgen.geom2d import unit_square
import time
import os

mesh = Mesh(unit_square.GenerateMesh(maxh=0.01))
fes = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(x*y*v*dx).Assemble()

blocks = fes.CreateSmoothingBlocks()
pre    = a.mat.CreateBlockSmoother(blocks)
adev   = a.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()

solver = ngscuda.DevCGSolver(
    adev, predev,
    adev_raw=adev,
    cdev_raw=predev,
    precision=1e-10,
    maxsteps=1000,
    printrates=False)

gfu = GridFunction(fes)

# warm up:
solver.Mult(f.vec, gfu.vec)

# timed solve:
t0 = time.perf_counter()
solver.Mult(f.vec, gfu.vec)
t1 = time.perf_counter()

use_graph = os.environ.get("NO_CUDA_GRAPH", "0") == "0"
print(f"ndof       = {fes.ndof}")
print(f"|sol|      = {Norm(gfu.vec):.8e}")
print(f"use_graph  = {use_graph}")
print(f"elapsed    = {(t1-t0)*1000:.3f} ms")
