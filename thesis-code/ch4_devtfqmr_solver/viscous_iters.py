import ngsolve
from ngsolve import *
from ngsolve import la
from netgen.occ import unit_cube
from time import perf_counter

mesh = Mesh(unit_cube.GenerateMesh(maxh=0.05))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()

dt = 0.01
nu = 0.001
a  = BilinearForm(fes)
a += (1/dt) * u*v*dx
a += nu * grad(u)*grad(v)*dx
a.Assemble()

blocks = fes.CreateSmoothingBlocks(blocktype="element")
pre    = a.mat.CreateBlockSmoother(blocks)
f      = LinearForm(1*v*dx).Assemble()
gfu    = GridFunction(fes)

import ngsolve.ngscuda as ngscuda
fdev   = f.vec.CreateDeviceVector(copy=True)
adev   = a.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()

print(f"ndof = {fes.ndof}")

# CG
solver_cg = la.CGSolver(adev, predev,
                         precision=1e-12,
                         maxsteps=400,
                         printrates=False)
gfu.vec.data = solver_cg * fdev
print(f"CG  iters: {solver_cg.GetSteps()}  |sol| = {Norm(gfu.vec):.8e}")

# GMRES  
solver_gm = la.GMRESSolver(adev, predev,
                            precision=1e-12,
                            maxsteps=400,
                            printrates=False)
gfu.vec.data = solver_gm * fdev
print(f"GMRES iters: {solver_gm.GetSteps()}  |sol| = {Norm(gfu.vec):.8e}")
