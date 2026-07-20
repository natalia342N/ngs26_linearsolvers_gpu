from ngsolve import *
from netgen.occ import unit_square
import ngsolve.ngscuda as ngscuda

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
m = BilinearForm(u*v*dx).Assemble()
pre = a.mat.CreateSmoother(fes.FreeDofs())

adev   = a.mat.CreateDeviceMatrix()
mdev   = m.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()

NUM = 4
solver = ngscuda.DevLOBPCGSolver(a=adev, m=mdev, pre=predev, num=NUM, maxit=100)
lams, evecs = solver.Solve()

print("\nGPU LOBPCG eigenvalues:")
for i, lam in enumerate(lams):
    print(f"  lambda_{i+1} = {lam:.8f}")

import math
expected = sorted(math.pi**2 * (i**2 + j**2) for i in range(1,5) for j in range(1,5))[:NUM]
print("\nExpected (analytical):")
for i, lam in enumerate(expected):
    print(f"  lambda_{i+1} = {lam:.8f}")
