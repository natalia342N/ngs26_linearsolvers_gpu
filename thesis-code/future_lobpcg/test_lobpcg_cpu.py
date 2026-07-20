from ngsolve import *
from netgen.occ import unit_square
from ngsolve.solvers import LOBPCG

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
m = BilinearForm(u*v*dx).Assemble()

pre = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")

# Find first 4 eigenvalues of  A u = lambda M u
NUM = 4
lams, vecs = LOBPCG(a.mat, m.mat, pre, num=NUM, maxit=30, printrates=True)

print("\nComputed eigenvalues:")
for i, lam in enumerate(lams):
    print(f"  lambda_{i+1} = {lam:.8f}")

# Analytical values for Laplacian on unit square: (pi^2*(i^2+j^2)) for i,j>=1
import math
expected = sorted(math.pi**2 * (i**2 + j**2) for i in range(1, 5) for j in range(1, 5))[:NUM]
print("\nExpected (analytical):")
for i, lam in enumerate(expected):
    print(f"  lambda_{i+1} = {lam:.8f}")
