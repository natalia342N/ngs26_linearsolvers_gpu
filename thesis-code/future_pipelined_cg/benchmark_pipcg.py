"""
Timing benchmark: DevCGSolver vs DevPCGSolver (pipelined CG with fused triple-dot).
Four problem sizes, 3D Poisson, H1 order 2, all-Dirichlet BCs.
"""
import time
from ngsolve import *
from netgen.csg import unit_cube
import ngsolve.ngscuda as ngscuda

sizes = [0.20, 0.15, 0.10, 0.07]
RUNS  = 5  # timed runs per solver per size

print(f"{'ndof':>8}  {'CG iters':>9}  {'CG ms':>8}  {'PipCG iters':>12}  {'PipCG ms':>10}  {'speedup':>8}")
print("-" * 72)

for maxh in sizes:
    mesh = Mesh(unit_cube.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=2, dirichlet=".*")
    ndof = fes.ndof

    u, v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    f = LinearForm(x*y*v*dx).Assemble()
    gfu = GridFunction(fes)

    pre = Projector(fes.FreeDofs(), True)

    adev   = a.mat.CreateDeviceMatrix()
    predev = pre.CreateDeviceMatrix()
    fdev   = f.vec.CreateDeviceVector(copy=True)

    cg_solver = ngscuda.DevCGSolver(mat=adev, pre=predev,
                                     adev_raw=adev, cdev_raw=predev,
                                     precision=1e-8, maxsteps=2000, printrates=False)
    pip_solver = ngscuda.DevPCGSolver(mat=adev, pre=predev,
                                      adev_raw=adev, cdev_raw=predev,
                                      precision=1e-8, maxsteps=2000, printrates=False)

    # Warm-up
    cg_solver.Mult(fdev, gfu.vec)
    pip_solver.Mult(fdev, gfu.vec)

    # Time DevCGSolver
    t0 = time.perf_counter()
    for _ in range(RUNS):
        cg_solver.Mult(fdev, gfu.vec)
    cg_ms = (time.perf_counter() - t0) / RUNS * 1000
    cg_iters = cg_solver.GetSteps()

    # Time DevPCGSolver
    t0 = time.perf_counter()
    for _ in range(RUNS):
        pip_solver.Mult(fdev, gfu.vec)
    pip_ms = (time.perf_counter() - t0) / RUNS * 1000
    pip_iters = pip_solver.GetSteps()

    speedup = cg_ms / pip_ms
    print(f"{ndof:>8}  {cg_iters:>9}  {cg_ms:>8.2f}  {pip_iters:>12}  {pip_ms:>10.2f}  {speedup:>8.3f}x")
