from ngsolve import *
from netgen.occ import unit_square, unit_cube
from ngsolve.solvers import LOBPCG
import ngsolve.ngscuda as ngscuda
from time import time

NUM   = 4
MAXIT = 100
NRUNS = 3

print("=== 2D problems ===")
for maxh in [0.05, 0.02, 0.01]:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a   = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    m   = BilinearForm(u*v*dx).Assemble()
    pre = a.mat.CreateSmoother(fes.FreeDofs())

    times = []
    for _ in range(NRUNS):
        t0 = time(); LOBPCG(a.mat, m.mat, pre, num=NUM, maxit=MAXIT, printrates=False); times.append(time()-t0)
    t_cpu = min(times)

    adev = a.mat.CreateDeviceMatrix(); mdev = m.mat.CreateDeviceMatrix(); predev = pre.CreateDeviceMatrix()
    times = []
    for _ in range(NRUNS):
        solver = ngscuda.DevLOBPCGSolver(a=adev, m=mdev, pre=predev, num=NUM, maxit=MAXIT)
        t0 = time(); solver.Solve(); times.append(time()-t0)
    t_gpu = min(times)

    print(f"  ndof={fes.ndof:7d}  CPU={t_cpu*1000:7.1f}ms  GPU={t_gpu*1000:7.1f}ms  speedup={t_cpu/t_gpu:.2f}x")

print("\n=== 3D problems ===")
for maxh in [0.15, 0.10, 0.07]:
    mesh = Mesh(unit_cube.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a   = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    m   = BilinearForm(u*v*dx).Assemble()
    pre = a.mat.CreateSmoother(fes.FreeDofs())

    times = []
    for _ in range(NRUNS):
        t0 = time(); LOBPCG(a.mat, m.mat, pre, num=NUM, maxit=MAXIT, printrates=False); times.append(time()-t0)
    t_cpu = min(times)

    adev = a.mat.CreateDeviceMatrix(); mdev = m.mat.CreateDeviceMatrix(); predev = pre.CreateDeviceMatrix()
    times = []
    for _ in range(NRUNS):
        solver = ngscuda.DevLOBPCGSolver(a=adev, m=mdev, pre=predev, num=NUM, maxit=MAXIT)
        t0 = time(); solver.Solve(); times.append(time()-t0)
    t_gpu = min(times)

    print(f"  ndof={fes.ndof:7d}  CPU={t_cpu*1000:7.1f}ms  GPU={t_gpu*1000:7.1f}ms  speedup={t_cpu/t_gpu:.2f}x")
