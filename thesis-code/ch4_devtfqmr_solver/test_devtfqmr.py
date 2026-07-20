import ngsolve
print(f"NGSolve: {ngsolve.__version__}")
from ngsolve import *
from ngsolve.solvers import TFQMR
import ngsolve.ngscuda as ngscuda
from time import perf_counter

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
with TaskManager():
    a.Assemble()
blocks = fes.CreateSmoothingBlocks(blocktype="element")
pre    = a.mat.CreateBlockSmoother(blocks)
pre1   = Projector(fes.FreeDofs(), True)
f      = LinearForm(1*v*dx).Assemble()
gfu    = GridFunction(fes)
tol    = 1e-12

fdev    = f.vec.CreateDeviceVector(copy=True)
adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()
rhs_dev = (predev * fdev).Evaluate()

# Python GPU TFQMR
t0 = perf_counter()
gfu.vec.data = TFQMR(mat=predev@adev, pre=pre1dev, rhs=rhs_dev,
                     maxsteps=400, printrates=False, tol=tol)
t1 = perf_counter()
print(f"Python GPU TFQMR:   |sol| = {Norm(gfu.vec):.8e}  time = {t1-t0:.3f}s")

# C++ DevTFQMRSolver
solver = ngscuda.DevTFQMRSolver(mat=predev@adev, pre=pre1dev,
                                 precision=tol, maxsteps=400)
t0 = perf_counter()
gfu.vec.data = solver * rhs_dev
t1 = perf_counter()
print(f"C++ DevTFQMRSolver: |sol| = {Norm(gfu.vec):.8e}  time = {t1-t0:.3f}s")
print(f"expected ~8.79147329e+00")
