"""
test_cg_nsys_cpp_dev.py
=======================
NSys capture: C++ CGSolver (ngsolve.la) on device matrices.
This is the tutorial 5.5.1 GPU baseline — C++ loop, no Python overhead,
but individual kernel launches and D2H convergence check each iteration.
"""
import ctypes
import ngsolve
from ngsolve import *
from netgen.occ import unit_square

cuda   = ctypes.CDLL("libcuda.so")
cudart = ctypes.CDLL("libcudart.so.12")

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
gfu = GridFunction(fes)

# zero DeviceVectors — copy from the zero-initialized gfu.vec
sol_warm     = gfu.vec.CreateDeviceVector(copy=True)
sol_profiled = gfu.vec.CreateDeviceVector(copy=True)

# Warmup: full solve so CUDA handles are warm before profiling
solver_warm = CGSolver(adev, jdev, maxsteps=1000, precision=1e-10, printrates=False)
solver_warm.Mult(fdev, sol_warm)
print(f"warmup iters = {solver_warm.GetSteps()}")

# Profiled run: 20 iterations — sol_profiled starts at zero, so CG must actually iterate
solver = CGSolver(adev, jdev, maxsteps=20, precision=1e-30, printrates=False)

cuda.cuProfilerStart()
cudart.cudaProfilerStart()
solver.Mult(fdev, sol_profiled)
cudart.cudaProfilerStop()
cuda.cuProfilerStop()

print(f"profiled: {solver.GetSteps()} iterations (C++ CGSolver, device matrices)")
