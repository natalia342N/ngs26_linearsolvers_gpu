"""
ns_run.py  (GPU cluster job)

Runs 2D Schäfer-Turek NS benchmark with GPU DevCGSolver and saves
the mesh + velocity field as ns_flow_data.npz for local plotting.
"""

import numpy as np
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from NavierStokesIPCS_MCS import NavierStokes
from netgen.geom2d import SplineGeometry

# ── geometry ───────────────────────────────────────────────────────────────────
geo = SplineGeometry()
geo.AddRectangle((0, 0), (2.2, 0.41),
                 bcs=["bottom", "outlet", "top", "inlet"])
geo.AddCircle((0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl")
mesh = Mesh(geo.GenerateMesh(maxh=0.04))
print(f"2D mesh: {mesh.GetNE(VOL)} elements, {sum(1 for _ in mesh.vertices)} vertices")

# ── parameters ─────────────────────────────────────────────────────────────────
nu        = 1e-3
Umax      = 1.5
timestep  = 2e-3
N_STEPS   = 80
uin       = (4 * Umax * y * (0.41 - y) / 0.41**2, 0)

# ── solver ─────────────────────────────────────────────────────────────────────
ns = NavierStokes(mesh, nu=nu, inflow="inlet", outflow="outlet",
                  wall="top|bottom|cyl", uin=uin, timestep=timestep,
                  order=2, verbose=1, use_gpu=True)

print("Solving initial condition …")
with TaskManager():
    ns.SolveInitial()

print(f"Running {N_STEPS} time steps …")
with TaskManager():
    for k in range(N_STEPS):
        ns.DoTimeStep()
        if (k + 1) % 20 == 0:
            print(f"  step {k+1}/{N_STEPS}  t = {(k+1)*timestep:.3f}")

print("Extracting field data …")

# project to nodal H1 for easy export
fes_plot = H1(mesh, order=1)
gf_ux = GridFunction(fes_plot)
gf_uy = GridFunction(fes_plot)
gf_ux.Set(ns.velocity[0])
gf_uy.Set(ns.velocity[1])

vx = gf_ux.vec.FV().NumPy().copy()
vy = gf_uy.vec.FV().NumPy().copy()

pts  = np.array([[v.point[0], v.point[1]] for v in mesh.vertices])
tris = np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)])

np.savez("ns_flow_data.npz",
         pts=pts, tris=tris, vx=vx, vy=vy,
         Umax=Umax, nu=nu, t_final=N_STEPS * timestep)

print("Saved ns_flow_data.npz")

# ── standalone HTML via ngsolve webgui (no browser/display needed) ─────────────
from ngsolve.webgui import Draw
vel_mag = sqrt(ns.velocity * ns.velocity)
Draw(vel_mag, mesh, filename="ns_velocity.html", min=0, max=Umax,
     settings={"Objects": {"Edges": False}})
print("Saved ns_velocity.html  (open in any browser)")
