import ngsolve
print(f"NGSolve: {ngsolve.__version__}")
from ngsolve import *
from ngsolve.solvers import GMRes, MinRes, QMR, TFQMR
from ngsolve import la
from netgen.occ import unit_cube
from time import perf_counter

printrates = False
tol        = 1e-12
maxsteps   = 400

# ── Problem: simplified Part 2 (mass + viscous stokes, symmetric SPD) ──
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
pre1   = Projector(fes.FreeDofs(), True)
f      = LinearForm(1*v*dx).Assemble()
gfu    = GridFunction(fes)

print(f"ndof = {fes.ndof}")
print(f"numthreads = {ngsglobals.numthreads}")

# ── CPU reference ───────────────────────────────────────────
print("\n── CPU ─────────────────────────────────────────────────")

with TaskManager():
    ts = perf_counter()
    gfu.vec.data = la.CGSolver(a.mat, pre,
                               precision=tol,
                               maxsteps=maxsteps,
                               printrates=printrates) * f.vec
    te = perf_counter()
    sol_ref = Norm(gfu.vec)
    print(f"C++ CGSolver:    |sol| = {sol_ref:.8e}  time = {te-ts:.3f}s")

with TaskManager():
    ts = perf_counter()
    gfu.vec.data = la.GMRESSolver(a.mat, pre,
                                  precision=tol,
                                  maxsteps=maxsteps,
                                  printrates=printrates) * f.vec
    te = perf_counter()
    print(f"C++ GMRESSolver: |sol| = {Norm(gfu.vec):.8e}  time = {te-ts:.3f}s")

with TaskManager():
    ts = perf_counter()
    gfu.vec.data = MinRes(mat=a.mat, pre=pre,
                          rhs=f.vec,
                          maxsteps=maxsteps,
                          printrates=printrates,
                          tol=tol)
    te = perf_counter()
    print(f"Python MinRes:   |sol| = {Norm(gfu.vec):.8e}  time = {te-ts:.3f}s")

# ── GPU ─────────────────────────────────────────────────────
print("\n── GPU ─────────────────────────────────────────────────")
import ngsolve.ngscuda as ngscuda

fdev    = f.vec.CreateDeviceVector(copy=True)
adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()

print(f"adev type:    {type(adev)}")
print(f"predev type:  {type(predev)}")

# C++ CGSolver on GPU — works via polymorphic dispatch, no graph
ts = perf_counter()
gfu.vec.data = la.CGSolver(adev, predev,
                            precision=tol,
                            maxsteps=maxsteps,
                            printrates=printrates) * fdev
te = perf_counter()
print(f"\nC++ CGSolver GPU:    |sol| = {Norm(gfu.vec):.8e}  time = {te-ts:.3f}s")

# C++ GMRESSolver on GPU
ts = perf_counter()
gfu.vec.data = la.GMRESSolver(adev, predev,
                               precision=tol,
                               maxsteps=maxsteps,
                               printrates=printrates) * fdev
te = perf_counter()
print(f"C++ GMRESSolver GPU: |sol| = {Norm(gfu.vec):.8e}  time = {te-ts:.3f}s")

# Python MinRes on GPU
ts = perf_counter()
gfu.vec.data = MinRes(mat=adev, pre=predev,
                      rhs=fdev,
                      maxsteps=maxsteps,
                      printrates=printrates,
                      tol=tol)
te = perf_counter()
print(f"Python MinRes GPU:   |sol| = {Norm(gfu.vec):.8e}  time = {te-ts:.3f}s")

# Python TFQMR on GPU (same as Part 1 baseline — for comparison)
rhs_dev = (predev * fdev).Evaluate()
ts = perf_counter()
gfu.vec.data = TFQMR(mat=predev@adev, pre=pre1dev,
                     rhs=rhs_dev,
                     maxsteps=maxsteps,
                     printrates=printrates,
                     tol=tol)
te = perf_counter()
print(f"Python TFQMR GPU:    |sol| = {Norm(gfu.vec):.8e}  time = {te-ts:.3f}s")

print(f"\nReference: {sol_ref:.8e}")
print("Expected: C++ CGSolver GPU ≈ C++ CGSolver CPU solution")
