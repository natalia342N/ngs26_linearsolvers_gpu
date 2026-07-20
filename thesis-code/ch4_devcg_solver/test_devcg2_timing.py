from ngsolve import *
import ngsolve.ngscuda as ngscuda
from ngsolve.ngscuda import *
import numpy as np
import os
from time import time

mesh = Mesh(unit_cube.GenerateMesh(maxh=0.05))
fes = H1(mesh, order=2, dirichlet=".*")
print("ndof =", fes.ndof)

u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(1*v*dx).Assemble()

pre = a.mat.CreateBlockSmoother(fes.CreateSmoothingBlocks(blocktype="vertex"))

fdev  = f.vec.CreateDeviceVector(copy=True)
adev  = a.mat.CreateDeviceMatrix()
predev = pre.CreateDeviceMatrix()

NRUNS = 5
gfu = GridFunction(fes)

def run_solver(use_graph, nruns):
    if not use_graph:
        os.environ["NO_CUDA_GRAPH"] = "1"
    elif "NO_CUDA_GRAPH" in os.environ:
        del os.environ["NO_CUDA_GRAPH"]

    times = []
    sol_norm = None
    for i in range(nruns):
        solver = ngscuda.DevCGSolver2(adev, predev,
                                      adev_raw=adev, cdev_raw=predev,
                                      precision=1e-10, maxsteps=1000,
                                      printrates=False)
        ts = time()
        gfu.vec.data = solver * f.vec
        te = time()
        times.append(te - ts)
        if i == 0:
            sol_norm = Norm(gfu.vec)
    return times, sol_norm

print("\n--- no-graph ---")
times_ng, sol_ng = run_solver(use_graph=False, nruns=NRUNS)
print(f"  |sol| = {sol_ng:.8e}")
print(f"  times: {[f'{t:.4f}' for t in times_ng]}")
print(f"  mean={np.mean(times_ng):.4f}s  min={np.min(times_ng):.4f}s")

print("\n--- graph ---")
times_g, sol_g = run_solver(use_graph=True, nruns=NRUNS)
print(f"  |sol| = {sol_g:.8e}")
print(f"  times: {[f'{t:.4f}' for t in times_g]}")
print(f"  mean={np.mean(times_g):.4f}s  min={np.min(times_g):.4f}s")

print(f"\nSpeedup (mean): {np.mean(times_ng)/np.mean(times_g):.2f}x")
print(f"Speedup (min):  {np.min(times_ng)/np.min(times_g):.2f}x")
print(f"Solutions match: {abs(sol_ng - sol_g) < 1e-10}")
