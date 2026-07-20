from ngsolve import *
from netgen.occ import unit_square
import ngsolve.ngscuda as ngscuda
from ngsolve.krylovspace import TFQMR, GMRes

mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
fes  = H1(mesh, order=1, dirichlet=".*")
u, v = fes.TnT()

eps = 1.0
wind = CF((1, 0))
a  = BilinearForm((eps*grad(u)*grad(v) + wind*grad(u)*v)*dx).Assemble()
f  = LinearForm(1*v*dx).Assemble()

# Jacobi smoother — GPU-compatible unlike sparse Cholesky
pre  = a.mat.CreateSmoother(fes.FreeDofs())
pre1 = Projector(fes.FreeDofs(), True)

print(f"ndof = {fes.ndof}")

# GMRes reference
gfu_gmres = GridFunction(fes)
gfu_gmres.vec.data = GMRes(A=a.mat, pre=pre, b=f.vec, maxsteps=400, printrates=False)
print(f"GMRes solution norm:     {Norm(gfu_gmres.vec):.8f}")

# CPU TFQMR (left-preconditioned: mat=pre@A, pre=identity, rhs=pre*f)
gfu_cpu = GridFunction(fes)
gfu_cpu.vec.data = TFQMR(mat=pre@a.mat, pre=pre1,
                          rhs=(pre*f.vec).Evaluate(),
                          maxsteps=400, printrates=False, tol=1e-8)
print(f"CPU TFQMR solution norm: {Norm(gfu_cpu.vec):.8f}")

diff = gfu_gmres.vec.CreateVector()
diff.data = gfu_gmres.vec - gfu_cpu.vec
print(f"GMRes vs CPU TFQMR diff: {Norm(diff):.2e}")

# GPU TFQMR — predev@adev as system matrix, pre1dev as trivial preconditioner
adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()
fdev    = f.vec.CreateDeviceVector(copy=True)

pa_dev   = predev @ adev               # GPU: pre*A product matrix
rhs_pre  = (predev * fdev).Evaluate()  # GPU: pre*f

gfu_gpu = GridFunction(fes)
solver = ngscuda.DevTFQMRSolver(mat=pre@a.mat, pre=pre1,
                                 adev_raw=pa_dev, cdev_raw=pre1dev,
                                 precision=1e-8, maxsteps=400, printrates=False)
solver.Mult(rhs_pre, gfu_gpu.vec)
print(f"GPU TFQMR solution norm: {Norm(gfu_gpu.vec):.8f}")

diff.data = gfu_gmres.vec - gfu_gpu.vec
print(f"GMRes vs GPU TFQMR diff: {Norm(diff):.2e}")
