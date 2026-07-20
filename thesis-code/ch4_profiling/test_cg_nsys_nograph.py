"""
test_cg_nsys_nograph.py
=======================
NSys capture: no-graph CG, 20 iterations only.
Intended for the thesis timeline figure showing per-iteration DtoH stalls.

ndof = 46741 (unit_square, maxh=0.01, H1 order=2) — matches thesis prose.
"""
import ctypes
import ngsolve
from ngsolve import *
from netgen.occ import unit_square
import ngsolve.ngscuda as ngscuda
import os

os.environ["NO_CUDA_GRAPH"] = "1"

print(f"NGSolve {ngsolve.__version__}")

mesh = Mesh(unit_square.GenerateMesh(maxh=0.01))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(x*y*v*dx).Assemble()
print(f"ndof = {fes.ndof}")

blocks = fes.CreateSmoothingBlocks()
pre    = a.mat.CreateBlockSmoother(blocks)
adev   = a.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()
fdev   = f.vec.CreateDeviceVector(copy=True)
gfu    = GridFunction(fes)

# Warmup: full solve so graph/handles are warm
solver_warm = ngscuda.DevCGSolver(adev, predev,
                                   adev_raw=adev, cdev_raw=predev,
                                   precision=1e-10, maxsteps=1000, printrates=False)
solver_warm.Mult(fdev, gfu.vec)
print(f"warmup |sol| = {Norm(gfu.vec):.8e}  iters = {solver_warm.GetSteps()}")

# Profiled run: 20 iterations (won't converge — intentional, want repeating pattern visible)
solver = ngscuda.DevCGSolver(adev, predev,
                              adev_raw=adev, cdev_raw=predev,
                              precision=1e-30, maxsteps=20, printrates=False)

cuda = ctypes.CDLL("libcuda.so")
cuda.cuProfilerStart()
solver.Mult(fdev, gfu.vec)
cuda.cuProfilerStop()

print(f"profiled: {solver.GetSteps()} iterations captured (no-graph)")
