from time import time
import ctypes

from ngsolve import *
from netgen.csg import unit_cube
import ngsolve.ngscuda as ngscuda


print("Convection CUDA-Graph matvec test")
print("ngscuda:", ngscuda.__file__)
print("Has CudaGraph:", hasattr(ngscuda, "CudaGraph"))

if not hasattr(ngscuda, "CudaGraph"):
    raise RuntimeError("Your ngscuda build has no CudaGraph")

# IMPORTANT: avoid internal cudaDeviceSynchronize inside ngscuda ops
if hasattr(ngscuda, "SetSyncKernels"):
    ngscuda.SetSyncKernels(False)

def devsync():
    if hasattr(ngscuda, "DeviceSynchronize"):
        ngscuda.DeviceSynchronize()
    else:
        # fallback: force a sync via scalar reduction (rarely needed)
        pass


# ---------------- mesh / spaces ----------------
mesh = Mesh(unit_cube.GenerateMesh(maxh=0.1))
mesh.Refine()

fes_wind  = HDiv(mesh, order=2)
fes_u     = VectorL2(mesh, order=2, piola=True)
fes_facet = VectorFacetFESpace(mesh, order=2, dirichlet="inflow")
fes       = fes_wind * fes_u * fes_facet

w, u, uhat = fes.TrialFunction()
v          = fes_u.TestFunction()

embw, embu, embuhat = fes.embeddings
restw, restu, restuhat = fes.restrictions

# ---------------- operator ----------------
conv = BilinearForm(trialspace=fes, testspace=fes_u, nonlinear_matrix_free_bdb=True)
conv += Grad(v) * w * u * dx
conv.Assemble()
convop = conv.mat

diag = BaseVector(fes_facet.ndof)
diag[:] = 1.0
diag[~fes_facet.FreeDofs()] = 0.5
halfinflowbnd = DiagonalMatrix(diag)

traceop = ConvertOperator(fes_u, fes_facet)
convop  = convop @ (IdentityMatrix() + embuhat @ halfinflowbnd @ traceop @ restu)

print("\n--- HOST convop (structure) ---")
print(convop.GetOperatorInfo())

# ---------------- data ----------------
gfu = GridFunction(fes)
gfu.components[0].Set(CF((1, 0, 0)))
gfu.components[1].Set(CF((1, 0, 0)))

gfv = GridFunction(fes_u)
gfv.Set(CF((x, 1, 0)))

hv = BaseVector(fes_u.ndof)

devuvec = gfu.vec.CreateDeviceVector(copy=True)
devvvec = gfv.vec.CreateDeviceVector(copy=True)

devconv = convop.CreateDeviceMatrix()
devhv   = hv.CreateDeviceVector(copy=False)

print("\n--- DEVICE devconv (structure) ---")
print(devconv.GetOperatorInfo())

# ---------------- warmup ----------------
for _ in range(10):
    devhv.data = devconv * devuvec
devsync()

# ---------------- graph capture (capture ONLY what you want) ----------------
g = ngscuda.CudaGraph()
g.BeginCapture()
devhv.data = devconv * devuvec
g.EndCapture()

devsync()

# ---------------- nsys capture window control ----------------
# We’ll start/stop CUDA profiler so nsys can capture only this region.
cudart = ctypes.CDLL("libcudart.so")
cudart.cudaProfilerStart()

runs = 2000
t0 = time()
for _ in range(runs):
    g.Launch()
devsync()
t1 = time()

cudart.cudaProfilerStop()

print("\nGraph time per launch:", (t1 - t0) / runs)
print("Check InnerProduct:", InnerProduct(devhv, devvvec))
