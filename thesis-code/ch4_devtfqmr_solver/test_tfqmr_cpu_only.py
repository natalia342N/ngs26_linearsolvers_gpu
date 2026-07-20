from ngsolve import *
from netgen.occ import unit_square
from ngsolve.krylovspace import TFQMR, GMRes

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
fes  = H1(mesh, order=1, dirichlet=".*")
u, v = fes.TnT()

eps = 1.0
wind = CF((1, 0))
a = BilinearForm((eps*grad(u)*grad(v) + wind*grad(u)*v)*dx).Assemble()
f = LinearForm(1*v*dx).Assemble()
pre = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")
pre1 = Projector(fes.FreeDofs(), True)

print(f"ndof = {fes.ndof}")

# GMRes reference
gfu_gmres = GridFunction(fes)
gfu_gmres.vec.data = GMRes(A=a.mat, pre=pre, b=f.vec, maxsteps=400, printrates=False)
print(f"GMRes solution norm: {Norm(gfu_gmres.vec):.8f}")

# CPU TFQMR — correct calling convention: left-preconditioned system
gfu_cpu = GridFunction(fes)
gfu_cpu.vec.data = TFQMR(mat=pre@a.mat, pre=pre1,
                          rhs=(pre*f.vec).Evaluate(),
                          maxsteps=400, printrates=False, tol=1e-8)
print(f"CPU TFQMR solution norm: {Norm(gfu_cpu.vec):.8f}")

diff = gfu_gmres.vec.CreateVector()
diff.data = gfu_gmres.vec - gfu_cpu.vec
print(f"GMRes vs CPU TFQMR diff: {Norm(diff):.2e}")
