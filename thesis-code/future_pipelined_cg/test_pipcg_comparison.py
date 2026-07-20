"""
Comparison: Standard CG vs Pipelined CG (lecture notes formula), both unpreconditioned.

Standard CG (lecture notes):
  p0 = r0 = P*b  (project rhs to free dofs)
  alpha_i = <ri,ri> / <pi,Api>
  x_{i+1} = x_i + alpha_i * p_i
  r_{i+1} = r_i - alpha_i * A*p_i
  beta_i  = <r_{i+1},r_{i+1}> / <r_i,r_i>
  p_{i+1} = r_{i+1} + beta_i * p_i

Pipelined CG (lecture notes):
  p0 = r0 = P*b
  Compute Ap0, alpha_0, beta_0 before loop
  x_i  = x_{i-1} + alpha_{i-1} * p_{i-1}
  r_i  = r_{i-1} - alpha_{i-1} * Ap_{i-1}
  p_i  = r_i + beta_{i-1} * p_{i-1}
  Ap_i = A*p_i
  <Ap_i,Ap_i>, <p_i,Ap_i>, <r_i,r_i>   <- all three in one sync
  alpha_i = <r_i,r_i> / <p_i,Ap_i>
  beta_i  = (alpha_i^2 * <Ap_i,Ap_i> - <r_i,r_i>) / <r_i,r_i>

Implemented in pure numpy — demonstrates the numerical instability clearly.
"""
import numpy as np
import scipy.sparse as sp
from ngsolve import *
from ngsolve import la
from netgen.csg import unit_cube

mesh = Mesh(unit_cube.GenerateMesh(maxh=0.15))
fes  = H1(mesh, order=2, dirichlet=".*")
ndof = fes.ndof
print(f"ndof = {ndof}")

u, v = fes.TnT()
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm(x*y*v*dx).Assemble()
gfu = GridFunction(fes)

proj = Projector(fes.FreeDofs(), True)

# CPU reference with preconditioner
with TaskManager():
    s_ref = la.CGSolver(mat=a.mat, pre=proj, precision=1e-10, maxsteps=2000, printrates=False)
    gfu.vec.data = s_ref * f.vec
ref_norm = Norm(gfu.vec)
print(f"CPU CG (reference):   iters={s_ref.GetSteps():4d}  |sol|={ref_norm:.10e}")

# Extract sparse matrix and rhs as numpy/scipy
rows, cols, vals = a.mat.COO()
A_full = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
b_full = np.array(f.vec)

# Restrict to free dofs only — avoids contamination from constrained rows/cols
free     = np.array(fes.FreeDofs(), dtype=bool)
free_idx = np.where(free)[0]
A = A_full[np.ix_(free_idx, free_idx)].tocsr()
b = b_full[free_idx]
n = len(free_idx)
print(f"free dofs = {n}")

tol = 1e-8
maxsteps = 2000

# -----------------------------------------------------------------------
# Standard CG (lecture notes, no preconditioner, projected rhs)
# -----------------------------------------------------------------------
def standard_cg(A, b, tol, maxsteps):
    x = np.zeros(len(b))
    r = b.copy()
    p = r.copy()
    gamma = r @ r
    r0norm = gamma**0.5
    threshold = tol * r0norm

    residuals = [r0norm]
    for i in range(maxsteps):
        Ap = A @ p
        delta = p @ Ap
        alpha = gamma / delta
        x += alpha * p
        r -= alpha * Ap
        gamma_new = r @ r
        res = gamma_new**0.5
        residuals.append(res)
        if res <= threshold:
            return x, i+1, residuals
        beta = gamma_new / gamma
        gamma = gamma_new
        p = r + beta * p
    return x, maxsteps, residuals

# -----------------------------------------------------------------------
# Pipelined CG (lecture notes formula, no preconditioner, projected rhs)
# -----------------------------------------------------------------------
def pipelined_cg(A, b, tol, maxsteps):
    x = np.zeros(len(b))
    r = b.copy()
    p = r.copy()
    gamma = r @ r
    r0norm = gamma**0.5
    threshold = tol * r0norm

    # Initial step: compute Ap0, alpha_0, beta_0
    Ap = A @ p
    delta = p @ Ap
    eta   = Ap @ Ap
    alpha = gamma / delta
    beta  = (alpha*alpha * eta - gamma) / gamma

    residuals = [r0norm]
    for i in range(maxsteps):
        x  = x + alpha * p
        r  = r - alpha * Ap
        p  = r + beta * p
        Ap = A @ p

        # Three dot products in one go (the "pipeline" step)
        eta   = Ap @ Ap
        delta = p  @ Ap
        gamma = r  @ r

        res = gamma**0.5
        residuals.append(res)
        if res <= threshold:
            return x, i+1, residuals

        alpha = gamma / delta
        beta  = (alpha*alpha * eta - gamma) / gamma

    return x, maxsteps, residuals

# -----------------------------------------------------------------------
# Run and compare
# -----------------------------------------------------------------------
def embed(x_free):
    x_full = np.zeros(ndof)
    x_full[free_idx] = x_free
    return x_full

print()
print("Running Standard CG (unpreconditioned, free-dof submatrix)...")
x_std, iters_std, res_std = standard_cg(A, b, tol, maxsteps)
gfu.vec.FV().NumPy()[:] = embed(x_std)
norm_std = Norm(gfu.vec)
print(f"Standard CG:          iters={iters_std:4d}  |sol|={norm_std:.10e}  err={abs(norm_std-ref_norm):.2e}")

print()
print("Running Pipelined CG (lecture notes formula, unpreconditioned)...")
x_pip, iters_pip, res_pip = pipelined_cg(A, b, tol, maxsteps)
gfu.vec.FV().NumPy()[:] = embed(x_pip)
norm_pip = Norm(gfu.vec)
print(f"Pipelined CG:         iters={iters_pip:4d}  |sol|={norm_pip:.10e}  err={abs(norm_pip-ref_norm):.2e}")

# Show first 10 residuals of each to compare convergence behaviour
print()
print(f"{'iter':>5}  {'std res/r0':>12}  {'pip res/r0':>12}")
r0 = res_std[0]
for i in range(min(11, len(res_std), len(res_pip))):
    print(f"{i:5d}  {res_std[i]/r0:12.6f}  {res_pip[i]/r0:12.6f}")

print()
if iters_pip < maxsteps and abs(norm_pip - ref_norm) < 1e-5:
    print("Result: Pipelined CG CONVERGED — both methods agree.")
else:
    print("Result: Pipelined CG DIVERGED or hit max iters.")
    print("  => The recurrence beta=(alpha^2*eta-gamma)/gamma assumes exact A-conjugacy,")
    print("     which breaks down in floating-point over many iterations.")
    print("  => Formula designed for MPI (reduces AllReduce 2->1); on single GPU")
    print("     the sync cost is negligible, so the instability is not worth it.")
