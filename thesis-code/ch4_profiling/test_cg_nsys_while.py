"""
test_cg_nsys_while.py
=====================
NSys capture: WHILE graph CG, 20 iterations.
Matches test_cg_nsys_nograph.py for direct timeline comparison.

ndof = 46749 (unit_square, maxh=0.01, H1 order=2)
"""
import ctypes
import ngsolve
from ngsolve import *
from netgen.occ import unit_square
import ngsolve.ngscuda as ngscuda

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

# Warmup: full solve so graph is built and instantiated
solver_warm = ngscuda.DevCGSolver(adev, predev,
                                   adev_raw=adev, cdev_raw=predev,
                                   precision=1e-10, maxsteps=1000, printrates=False)
solver_warm.Mult(fdev, gfu.vec)
print(f"warmup |sol| = {Norm(gfu.vec):.8e}  iters = {solver_warm.GetSteps()}")

# Profiled run: exactly 20 iterations (won't converge — intentional, matches no-graph capture)
solver = ngscuda.DevCGSolver(adev, predev,
                              adev_raw=adev, cdev_raw=predev,
                              precision=1e-30, maxsteps=20, printrates=False)

cuda = ctypes.CDLL("libcuda.so")
cuda.cuProfilerStart()
solver.Mult(fdev, gfu.vec)
cuda.cuProfilerStop()

print(f"profiled: {solver.GetSteps()} iterations (WHILE graph)")
