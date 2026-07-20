import ngsolve
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from netgen.geom2d import unit_square
import ctypes
import os

os.environ["NO_CUDA_GRAPH"] = "1"   # disable graph

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
fes = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(x*y*v*dx).Assemble()

blocks = fes.CreateSmoothingBlocks()
pre    = a.mat.CreateBlockSmoother(blocks)
adev   = a.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()

solver2 = ngscuda.DevCGSolver2(
    adev, predev,
    adev_raw=adev,
    cdev_raw=predev,
    precision=1e-10,
    maxsteps=400,
    printrates=False)

gfu = GridFunction(fes)

# warm up:
solver2.Mult(f.vec, gfu.vec)

# profile second solve:
cudart = ctypes.CDLL("libcudart.so")
cudart.cudaProfilerStart()
solver2.Mult(f.vec, gfu.vec)
cudart.cudaProfilerStop()

print(f"|sol| = {Norm(gfu.vec):.8e}")
