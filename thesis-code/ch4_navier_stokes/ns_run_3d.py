"""
ns_run_3d.py  (GPU cluster job)

3D Schäfer-Turek benchmark with DevCGSolver (CudaWhileGraph).
Geometry matches the UM26 tutorial exactly.
Exports ns_velocity_3d.html — open in any browser, no server needed.
"""

import numpy as np
from ngsolve import *
from netgen.occ import *
import ngsolve.ngscuda as ngscuda
from NavierStokesIPCS_MCS import NavierStokes

order = 2

# ── geometry (exact UM26 tutorial geometry) ───────────────────────────────────
box = Box((0, 0, 0), (2.5, 0.41, 0.41))
box.faces.name = "wall"
box.faces.Min(X).name = "inlet"
box.faces.Max(X).name = "outlet"
cyl = Cylinder((0.5, 0.2, 0), Z, h=0.41, r=0.05)
cyl.faces.name = "cyl"
cyl.faces.maxh = 0.03
shape = box - cyl

print("Generating 3D mesh …")
mesh = Mesh(OCCGeometry(shape).GenerateMesh(maxh=0.1)).Curve(order)
print(f"3D mesh: {mesh.GetNE(VOL)} elements")

# ── parameters ────────────────────────────────────────────────────────────────
nu       = 1e-3
um       = 2.25
timestep = 2e-3 / um          # ≈ 8.89e-4  (matches UM26 tutorial)
N_STEPS  = 130                # t_final ≈ 0.116 s
uin      = (um * 16 * y * (0.41 - y) * z * (0.41 - z) / 0.41**4, 0, 0)

# ── solver ────────────────────────────────────────────────────────────────────
ns = NavierStokes(mesh, nu=nu, inflow="inlet", outflow="outlet",
                  wall="wall|cyl", uin=uin, timestep=timestep,
                  order=order, verbose=1, use_gpu=True)

print("Solving initial condition …")
with TaskManager():
    ns.SolveInitial()

print(f"Running {N_STEPS} time steps …")
with TaskManager():
    for k in range(N_STEPS):
        ns.DoTimeStep()
        if (k + 1) % 20 == 0:
            print(f"  step {k+1}/{N_STEPS}  t = {(k+1)*timestep:.4f}")

t_final = N_STEPS * timestep
print(f"Reached t = {t_final:.4f} s")

# ── export webgui HTML ────────────────────────────────────────────────────────
from ngsolve.webgui import Draw

clipping = {"function": True, "pnt": (2.5, 0.2, 0.2), "vec": (0, 0, -1)}
Draw(ns.velocity, clipping=clipping, order=order,
     min=0, max=um, filename="ns_velocity_3d.html")
print("Saved ns_velocity_3d.html  (open in any browser)")
