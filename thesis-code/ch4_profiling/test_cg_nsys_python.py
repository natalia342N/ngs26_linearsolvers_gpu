"""
test_cg_nsys_python.py
======================
NSys capture: Python-level CGSolver (krylovspace.py) on device vectors.
Uses the pure Python CG loop with explicit InnerProduct calls —
each iteration forces a D2H transfer to return a scalar to Python.
"""
import ctypes
import ngsolve
from ngsolve import *
from ngsolve.krylovspace import CGSolver as PyCGSolver
from netgen.occ import unit_square

print(f"NGSolve {ngsolve.__version__}")

mesh = Mesh(unit_square.GenerateMesh(maxh=0.01))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(x*y*v*dx).Assemble()
print(f"ndof = {fes.ndof}")

jac  = a.mat.CreateSmoother(fes.FreeDofs())
adev = a.mat.CreateDeviceMatrix()
jdev = jac.CreateDeviceMatrix()
fdev = f.vec.CreateDeviceVector(copy=True)

# Device solution vector — forces krylovspace to use device vectors throughout
sol_dev = fdev.CreateVector()

# Warmup: full solve so CUDA handles are warm before profiling
solver_warm = PyCGSolver(mat=adev, pre=jdev, maxsteps=1000, printrates=False, tol=1e-10)
solver_warm.Solve(rhs=fdev, sol=sol_dev)
print(f"warmup |sol| = {Norm(sol_dev):.8e}  iters = {solver_warm.iterations}")

# Profiled run: 20 iterations (tol=1e-30 so it runs all 20)
solver = PyCGSolver(mat=adev, pre=jdev, maxsteps=20, printrates=False, tol=1e-30)

cuda = ctypes.CDLL("libcuda.so")
cudart = ctypes.CDLL("libcudart.so.12")
cuda.cuProfilerStart()
cudart.cudaProfilerStart()

solver.Solve(rhs=fdev, sol=sol_dev)

cudart.cudaProfilerStop()
cuda.cuProfilerStop()

print(f"profiled: {solver.iterations} iterations (Python krylovspace CG, device vectors)")
