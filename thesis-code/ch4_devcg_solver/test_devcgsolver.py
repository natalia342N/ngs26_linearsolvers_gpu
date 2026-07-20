import ngsolve
print(f"NGSolve: {ngsolve.__version__}")
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from ngsolve import la
from netgen.occ import unit_cube
from time import perf_counter

mesh = Mesh(unit_cube.GenerateMesh(maxh=0.05))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()

dt = 0.01
nu = 0.001
a  = BilinearForm(fes)
a += (1/dt)*u*v*dx
a += nu*grad(u)*grad(v)*dx
a.Assemble()

blocks = fes.CreateSmoothingBlocks(blocktype="element")
pre    = a.mat.CreateBlockSmoother(blocks)
f      = LinearForm(1*v*dx).Assemble()
gfu    = GridFunction(fes)

fdev   = f.vec.CreateDeviceVector(copy=True)
adev   = a.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()

print(f"ndof = {fes.ndof}")

# reference
gfu.vec.data = la.CGSolver(adev, predev,
                            precision=1e-12,
                            maxsteps=400,
                            printrates=False) * fdev
sol_ref = Norm(gfu.vec)
print(f"C++ CGSolver ref:  |sol| = {sol_ref:.8e}")

# DevCGSolver use_graph=1
t0 = perf_counter()
solver = ngscuda.DevCGSolver(mat=adev, pre=predev,
                              adev_raw=adev,
                              cdev_raw=predev,
                              precision=1e-12,
                              maxsteps=400)
gfu.vec.data = solver * fdev
t1 = perf_counter()
print(f"DevCGSolver:       |sol| = {Norm(gfu.vec):.8e}  time = {t1-t0:.3f}s")
print(f"expected:                  {sol_ref:.8e}")
print(f"match: {abs(Norm(gfu.vec) - sol_ref) < 1e-4}")

# DevCGSolver use_graph=0
import os
os.environ['NO_CUDA_GRAPH'] = '1'
t0 = perf_counter()
solver2 = ngscuda.DevCGSolver(mat=adev, pre=predev,
                               adev_raw=adev,
                               cdev_raw=predev,
                               precision=1e-12,
                               maxsteps=400)
gfu.vec.data = solver2 * fdev
t1 = perf_counter()
print(f"DevCGSolver nogph: |sol| = {Norm(gfu.vec):.8e}  time = {t1-t0:.3f}s")

print(f"\npredev type: {type(predev)}")
print(f"predev.__class__.__mro__: {predev.__class__.__mro__}")
