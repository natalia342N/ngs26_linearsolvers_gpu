import ngsolve
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from netgen.geom2d import unit_square
import time
import os

use_graph = os.environ.get("NO_CUDA_GRAPH", "0") == "0"

for maxh in [0.8, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    f = LinearForm(x*y*v*dx).Assemble()

    blocks = fes.CreateSmoothingBlocks()
    pre    = a.mat.CreateBlockSmoother(blocks)
    adev   = a.mat.CreateDeviceMatrix()
    predev = pre.CreateDeviceMatrix()

    solver = ngscuda.DevCGSolver(
        adev, predev,
        adev_raw=adev,
        cdev_raw=predev,
        precision=1e-10,
        maxsteps=1000,
        printrates=False)

    gfu = GridFunction(fes)

    # warm up
    solver.Mult(f.vec, gfu.vec)
    solver.Mult(f.vec, gfu.vec)

    # average over 3 runs
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        solver.Mult(f.vec, gfu.vec)
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)
    avg = sum(times)/len(times)

    print(f"maxh={maxh:.3f}  ndof={fes.ndof:8d}  use_graph={use_graph}  elapsed={avg:.3f} ms")
