"""
CPU-only CG benchmark — run locally without CUDA.
Same problem as benchmark_cg_jacobi_scaling.py: unit square, H1 order 2, Poisson, Jacobi smoother.
"""
import time
from ngsolve import *
from netgen.geom2d import unit_square

RUNS = 3

print(f"CPU threads (NGSolve TaskManager): {ngsglobals.numthreads}")
print(f"{'ndof':>10}  {'CPU CG (ms)':>12}")
print("-" * 28)

for maxh in [0.8, 0.5, 0.3, 0.15, 0.07, 0.03, 0.01, 0.005]:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()

    with TaskManager():
        a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    f = LinearForm(x*y*v*dx).Assemble()

    jac = a.mat.CreateSmoother(fes.FreeDofs())

    gfu = GridFunction(fes)

    times = []
    for _ in range(RUNS):
        gfu.vec[:] = 0.0
        with TaskManager():
            t0 = time.perf_counter()
            gfu.vec.data = CGSolver(a.mat, jac, maxsteps=10000, precision=1e-10) * f.vec
            t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    ms = sorted(times)[RUNS // 2]
    print(f"{fes.ndof:>10}  {ms:>12.1f}")
