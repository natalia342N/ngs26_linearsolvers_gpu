from ngsolve import *
from time import time
import ngsolve.ngscuda as ngscuda

tol = 1e-12

# -------------------------
# Build problem (CPU)
# -------------------------
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.05))
fes = L2(mesh, order=2, dgjumps=True)
print("ndof =", fes.ndof)

u, v = fes.TnT()
wind = CF((1, 0.2, 0.3))
n = specialcf.normal(mesh.dim)
dS = dx(element_boundary=True)

a = BilinearForm(fes)
a += -20*u*v*dx
a += u*wind*grad(v)*dx
a += -(wind*n)*IfPos(wind*n, u, u.Other(bnd=0))*v*dS
with TaskManager():
    a.Assemble()

blocks = fes.CreateSmoothingBlocks(blocktype="element")
pre = a.mat.CreateBlockSmoother(blocks)
pre1 = Projector(fes.FreeDofs(), True)

f = LinearForm(1*v*dx).Assemble()

print("On CPU, numthreads =", ngsglobals.numthreads)
print("on Device:")

# -------------------------
# Device operators
# -------------------------
fdev    = f.vec.CreateDeviceVector(copy=True)
adev    = a.mat.CreateDeviceMatrix()
predev  = pre.CreateDeviceMatrix()
pre1dev = pre1.CreateDeviceMatrix()

# -------------------------
# Preallocate device vectors (addresses must stay fixed)
# -------------------------
x = fdev.CreateVector()   # current iterate / input
y = fdev.CreateVector()   # temp: A*x
z = fdev.CreateVector()   # temp: P^{-1} y

# initialize x deterministically
x.data = fdev

alpha = 0.1

def sync_vec(vec):
    # no DeviceSynchronize in your build; Norm forces GPU + scalar return
    _ = float(Norm(vec))

# -------------------------
# Warm-up: run the pipeline a few times (no graph)
# (forces lazy init, workspace alloc, module loads)
# -------------------------
for _ in range(5):
    y.data = adev * x
    y.data = (predev * y).Evaluate()   # may create a temp once; OK outside capture
    z.data = pre1dev * y
    x.data += alpha * z
sync_vec(x)

# -------------------------
# Baseline timing (no graph)
# -------------------------
runs = 200
t0 = time()
for _ in range(runs):
    y.data = adev * x
    y.data = (predev * y).Evaluate()
    z.data = pre1dev * y
    x.data += alpha * z
sync_vec(x)
t1 = time()
print("baseline time/step =", (t1 - t0) / runs)

# -------------------------
# Graph capture of ONE step
# IMPORTANT: avoid Evaluate() if it allocates during capture.
# If this fails, we replace it with a graph-safe variant (see notes below).
# -------------------------
g = ngscuda.CudaGraph()

try:
    g.BeginCapture()
    y.data = adev * x
    y.data = (predev * y).Evaluate()
    z.data = pre1dev * y
    x.data += alpha * z
    g.EndCapture()
    sync_vec(x)
    print("Capture succeeded.")
except Exception as e:
    print("Capture failed:", repr(e))
    raise

# -------------------------
# Replay timing
# -------------------------
t0 = time()
for _ in range(runs):
    g.Launch()
sync_vec(x)
t1 = time()
print("graph time/step =", (t1 - t0) / runs)
