import ctypes, ngsolve
print(f"NGSolve: {ngsolve.__version__}")
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from netgen.occ import unit_cube

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

blocks  = fes.CreateSmoothingBlocks(blocktype="element")
pre     = a.mat.CreateBlockSmoother(blocks)
pre1    = Projector(fes.FreeDofs(), True)
f       = LinearForm(1*v*dx).Assemble()
gfu     = GridFunction(fes)

fdev    = f.vec.CreateDeviceVector(copy=True)
adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()
rhs_dev = (predev * fdev).Evaluate()

solver = ngscuda.DevTFQMRSolver(mat=predev@adev, pre=pre1dev,
                                 adev_raw=adev, cdev_raw=predev, ddev_raw=pre1dev,
                                 precision=1e-12, maxsteps=400)

cuda = ctypes.CDLL("libcuda.so")

# warm up (graph already built during DevTFQMRSolver construction)
gfu.vec.data = solver * rhs_dev
print(f"warm-up |sol| = {Norm(gfu.vec):.8e}")

# profiled solve — only this region appears in Nsight Systems
cuda.cuProfilerStart()
gfu.vec.data = solver * rhs_dev
cuda.cuProfilerStop()

print(f"profiled |sol| = {Norm(gfu.vec):.8e}")
print(f"expected ~8.79147329e+00")
