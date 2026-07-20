"""
benchmark_ns_timing.py

Compares three execution modes for the Navier-Stokes IPCS velocity solve:
  1. CPU    — CGSolver with BDDC preconditioner (reference)
  2. no-graph GPU — la.CGSolver on device matrices, projector preconditioner
  3. graph GPU  — DevCGSolver (CudaWhileGraph), projector preconditioner

Modes 2 and 3 use the same preconditioner so their comparison is fair.
Mode 1 uses BDDC so its iteration count differs; shown as a CPU reference only.

Runs WARMUP time steps (discarded), then TIMED time steps and reports
mean ms/step for the velocity solve only.
"""

from ngsolve import *
from ngsolve import la
from ngsolve.krylovspace import QMRSolver
import ngsolve.ngscuda as ngscuda
from netgen.occ import *
from time import perf_counter
import numpy as np

WARMUP = 2
TIMED  = 5

# ---- geometry (same as navierstokes_run.py) ----
box = Box((0,0,0), (2.5,0.41,0.41))
box.faces.name = "wall"
box.faces.Min(X).name = "inlet"
box.faces.Max(X).name = "outlet"
cyl = Cylinder((0.5,0.2,0), Z, h=0.41, r=0.05)
cyl.faces.name = "cyl"
cyl.faces.maxh = 0.03
shape = box - cyl

order    = 2
mesh     = Mesh(OCCGeometry(shape).GenerateMesh(maxh=0.1)).Curve(order)
um       = 2.25
timestep = 2e-3 / um
nu       = 1e-3
uin      = (um*16*y*(0.41-y)*z*(0.41-z)/0.41**4, 0, 0)

print(f"ne = {mesh.GetNE(VOL)}")

# ---- common FE spaces (shared across all modes) ----
V     = HDiv(mesh, order=order, dirichlet="inlet|wall|", highest_order_dc=True)
Vhat  = TangentialFacetFESpace(mesh, order=order-1, dirichlet="inlet|wall||outlet")
from ngsolve import HCurlDiv, MatrixValued
Sigma = Compress(PrivateSpace(Discontinuous(HCurlDiv(mesh, order=order-1, orderinner=order))))
S     = Compress(PrivateSpace(MatrixValued(L2(mesh, order=order-1), skewsymmetric=True)))
X     = V*Vhat*Sigma*S
print(f"ndof X = {X.ndof}")

u, uhat, sigma, W = X.TrialFunction()
v, vhat, tau,   R = X.TestFunction()
dS = dx(element_boundary=True, bonus_intorder=2)
n  = specialcf.normal(mesh.dim)
def tang(u): return u - (u*n)*n

stokesA = (-0.5/nu * InnerProduct(sigma,tau) * dx
           + (div(sigma)*v + div(tau)*u) * dx
           + (InnerProduct(W,tau) + InnerProduct(R,sigma)) * dx
           - (((sigma*n)*n)*(v*n) + ((tau*n)*n)*(u*n)) * dS
           - ((sigma*n)*tang(vhat) + (tau*n)*tang(uhat)) * dS)

# Non-condensed mstar for GPU (device matrix needs non-condensed form)
mstar_nc = BilinearForm(X, eliminate_hidden=True)
mstar_nc += u*v*dx + timestep * stokesA
with TaskManager():
    mstar_nc.Assemble()

# Condensed mstar for CPU (BDDC needs condense=True)
mstar_cpu = BilinearForm(X, eliminate_hidden=True, condense=True)
mstar_cpu += u*v*dx + timestep * stokesA
pre_bddc  = preconditioners.BDDC(mstar_cpu)
with TaskManager():
    mstar_cpu.Assemble()

print("Assembled mstar (CPU condensed + GPU non-condensed)")

# ---- GPU operators (shared between no-graph and graph modes) ----
adev = mstar_nc.mat.CreateDeviceMatrix()
proj = Projector(X.FreeDofs(), True)
jdev = proj.CreateDeviceMatrix()
fdev_tmp = mstar_nc.mat.CreateColVector().CreateDeviceVector()

# ---- three solver objects ----
inv_cpu = la.CGSolver(mstar_cpu.mat, pre=pre_bddc,
                      precision=1e-7, maxsteps=2000, printrates=False)
ext_cpu  = IdentityMatrix() + mstar_cpu.harmonic_extension
extT_cpu = IdentityMatrix() + mstar_cpu.harmonic_extension_trans
inv_cpu_full = ext_cpu @ inv_cpu @ extT_cpu + mstar_cpu.inner_solve

inv_nograph = la.CGSolver(adev, jdev,
                          precision=1e-7, maxsteps=2000, printrates=False)

inv_graph = ngscuda.DevCGSolver(mat=adev, pre=jdev,
                                adev_raw=adev, cdev_raw=jdev,
                                precision=1e-7, maxsteps=2000,
                                printrates=False)

print("Solver objects ready.\n")

# ---- timing helper ----
def time_solve(label, solve_fn, n_steps):
    times = []
    for i in range(n_steps):
        rhs = mstar_nc.mat.CreateColVector()
        rhs[:] = np.random.randn(len(rhs))   # synthetic RHS each step
        sol = mstar_nc.mat.CreateColVector()
        t0 = perf_counter()
        solve_fn(rhs, sol)
        times.append((perf_counter() - t0) * 1000)
    return times

def solve_cpu(rhs, sol):
    sol.data = inv_cpu_full * rhs

def solve_nograph(rhs, sol):
    fdev = rhs.CreateDeviceVector(copy=True)
    sdev = sol.CreateDeviceVector()
    inv_nograph.Mult(fdev, sdev)
    sol.data = sdev

def solve_graph(rhs, sol):
    fdev = rhs.CreateDeviceVector(copy=True)
    sdev = sol.CreateDeviceVector()
    inv_graph.Mult(fdev, sdev)
    sol.data = sdev

print(f"=== Warm-up ({WARMUP} steps each) ===")
time_solve("cpu",     solve_cpu,     WARMUP)
time_solve("nograph", solve_nograph, WARMUP)
time_solve("graph",   solve_graph,   WARMUP)

print(f"=== Timed ({TIMED} steps each) ===")
t_cpu     = time_solve("cpu",     solve_cpu,     TIMED)
t_nograph = time_solve("nograph", solve_nograph, TIMED)
t_graph   = time_solve("graph",   solve_graph,   TIMED)

def stats(ts):
    return np.mean(ts), np.min(ts), np.max(ts)

mc, minc, maxc = stats(t_cpu)
mn, minn, maxn = stats(t_nograph)
mg, ming, maxg = stats(t_graph)

print(f"\n{'Solver':<30} {'mean(ms)':>10} {'min':>8} {'max':>8} {'speedup':>9}")
print("-" * 68)
print(f"{'CPU CGSolver (BDDC)':<30} {mc:10.1f} {minc:8.1f} {maxc:8.1f} {'—':>9}")
print(f"{'GPU no-graph (la.CGSolver)':<30} {mn:10.1f} {minn:8.1f} {maxn:8.1f} {mc/mn:8.2f}x")
print(f"{'GPU graph (DevCGSolver)':<30} {mg:10.1f} {ming:8.1f} {maxg:8.1f} {mn/mg:8.2f}x")
print(f"\nGraph vs no-graph speedup: {mn/mg:.2f}x")
print(f"Graph vs CPU speedup:      {mc/mg:.2f}x")
