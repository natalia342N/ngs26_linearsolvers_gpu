"""
benchmark_convergence.py

Compares per-step residual history between:
  - DevCGSolver NO_CUDA_GRAPH=1 (same impl, no graph, GPU device matrices)
  - DevCGSolver               (WHILE graph, GPU device matrices)

Both use identical C++ implementation — only execution mode differs.
GetResiduals() available on both paths after our C++ modification.
"""

from ngsolve import *
from ngsolve import la
import ngsolve.ngscuda as ngscuda
from netgen.geom2d import unit_square
import numpy as np
import os

# --- Problem 1: 2D Poisson, H1 order 2, point-Jacobi preconditioner ---
mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
fes  = H1(mesh, order=2, dirichlet=".*")
u, v = fes.TnT()
a    = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f    = LinearForm(1*v*dx).Assemble()
pre  = Preconditioner(a, "local")   # point Jacobi

print(f"ndof = {fes.ndof}")

# --- device objects ---
adev = a.mat.CreateDeviceMatrix()
jdev = pre.mat.CreateDeviceMatrix()
bdev = f.vec.CreateDeviceVector(copy=True)

# -----------------------------------------------------------------------
# Mode 1: DevCGSolver with NO_CUDA_GRAPH=1 (no graph, same implementation)
# -----------------------------------------------------------------------
print("\n[1] DevCGSolver NO_CUDA_GRAPH=1 (no graph)")
os.environ["NO_CUDA_GRAPH"] = "1"
inv_nograph = ngscuda.DevCGSolver(mat=adev, pre=jdev,
                                   adev_raw=adev, cdev_raw=jdev,
                                   precision=1e-8, maxsteps=2000, printrates=False)
xvec1 = f.vec.CreateDeviceVector()
inv_nograph.Mult(bdev, xvec1)
del os.environ["NO_CUDA_GRAPH"]

steps_nograph = inv_nograph.GetSteps()
res_nograph   = list(inv_nograph.GetResiduals())
print(f"  steps:     {steps_nograph}")
print(f"  residuals: {len(res_nograph)} values")
if res_nograph:
    print(f"  r[0]  = {res_nograph[0]:.4e}")
    print(f"  r[-1] = {res_nograph[-1]:.4e}")

# -----------------------------------------------------------------------
# Mode 2: DevCGSolver with WHILE graph
# -----------------------------------------------------------------------
print("\n[2] DevCGSolver (WHILE graph)")
inv_graph = ngscuda.DevCGSolver(mat=adev, pre=jdev,
                                 adev_raw=adev, cdev_raw=jdev,
                                 precision=1e-8, maxsteps=2000, printrates=False)
xvec2 = f.vec.CreateDeviceVector()
inv_graph.Mult(bdev, xvec2)

steps_graph = inv_graph.GetSteps()
res_graph   = list(inv_graph.GetResiduals())
print(f"  steps:     {steps_graph}")
print(f"  residuals: {len(res_graph)} values")
if res_graph:
    print(f"  r[0]  = {res_graph[0]:.4e}")
    print(f"  r[-1] = {res_graph[-1]:.4e}")

# -----------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------
print(f"\n=== Convergence comparison ===")
print(f"  DevCGSolver no-graph:  {steps_nograph:4d} iterations")
print(f"  DevCGSolver graph:     {steps_graph:4d} iterations")
print(f"  Counts match: {'YES' if steps_nograph == steps_graph else 'NO — differ by ' + str(abs(steps_nograph - steps_graph))}")

# -----------------------------------------------------------------------
# Save data
# -----------------------------------------------------------------------
if res_nograph:
    np.savetxt('residuals_nograph.txt', res_nograph,
               header=f'DevCGSolver no-graph  ndof={fes.ndof}  steps={steps_nograph}')
    print(f"\nSaved residuals_nograph.txt")

if res_graph:
    np.savetxt('residuals_graph.txt', res_graph,
               header=f'DevCGSolver WHILE graph  ndof={fes.ndof}  steps={steps_graph}')
    print(f"Saved residuals_graph.txt")

# -----------------------------------------------------------------------
# Mode 3: Pure Python CG on device matrices
# -----------------------------------------------------------------------
print("\n[3] Python CG (device matrices, no C++ solver)")

def python_cg_dev(A, P, b, tol=1e-8, maxsteps=2000):
    """Preconditioned CG in Python using device matrix operations."""
    x = b.CreateDeviceVector()          # zero initial guess
    r = b.CreateDeviceVector(copy=True) # r = b
    z = b.CreateDeviceVector()
    p = b.CreateDeviceVector()
    q = b.CreateDeviceVector()

    P.Mult(r, z)                        # z = P^{-1} r
    p.data = z                          # p = z
    rz = InnerProduct(r, z)
    r0 = (abs(rz))**0.5

    residuals = [(abs(rz))**0.5]
    steps = 0
    for k in range(maxsteps):
        A.Mult(p, q)                    # q = A p
        pq = InnerProduct(p, q)
        alpha = rz / pq
        x.Add(p, alpha)                 # x += alpha * p
        r.Add(q, -alpha)                # r -= alpha * q
        P.Mult(r, z)                    # z = P^{-1} r
        rz_new = InnerProduct(r, z)
        beta = rz_new / rz
        p.Add(p, beta - 1.0)           # p += (beta-1)*p  →  p = beta*p
        p.Add(z, 1.0)                   # p += z
        rz = rz_new
        res = (abs(rz))**0.5
        residuals.append(res)
        steps = k + 1
        if res <= tol * r0:
            break
    return x, residuals, steps

xvec3, res_python, steps_python = python_cg_dev(adev, jdev, bdev)
print(f"  steps:     {steps_python}")
print(f"  residuals: {len(res_python)} values")
if res_python:
    print(f"  r[0]  = {res_python[0]:.4e}")
    print(f"  r[-1] = {res_python[-1]:.4e}")

if res_python:
    np.savetxt('residuals_python.txt', res_python,
               header=f'Python CG dev matrices  ndof={fes.ndof}  steps={steps_python}')
    print(f"Saved residuals_python.txt")

# -----------------------------------------------------------------------
# Side-by-side residual table (all three modes)
# -----------------------------------------------------------------------
if res_nograph and res_graph and res_python:
    print(f"\n{'Iter':>5}  {'No-graph':>16}  {'WHILE graph':>16}  {'Python CG':>16}")
    print("-" * 60)
    n = max(len(res_nograph), len(res_graph), len(res_python))
    for i in range(n):
        ng = f"{res_nograph[i]:.6e}" if i < len(res_nograph) else "           —"
        g  = f"{res_graph[i]:.6e}"   if i < len(res_graph)   else "           —"
        py = f"{res_python[i]:.6e}"  if i < len(res_python)  else "           —"
        print(f"{i+1:>5}  {ng:>16}  {g:>16}  {py:>16}")
