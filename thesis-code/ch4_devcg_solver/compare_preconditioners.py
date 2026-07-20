"""
compare_preconditioners.py
==========================
Compare point-Jacobi vs block-Gauss-Seidel preconditioners
for DevCGSolver (3D Poisson) and DevTFQMRSolver (2D convection-diffusion).

Both preconditioner types have registered GPU implementations:
  - CreateSmoother  -> JacobiPrecond -> DevDiagonalMatrix
  - CreateBlockSmoother -> BlockDiagonalMatrixSoA -> DevBlockDiagonalMatrixSoA
"""
import ngsolve
from ngsolve import *
from netgen.csg import unit_cube
from netgen.occ import unit_square
import ngsolve.ngscuda as ngscuda
import time

NRUNS    = 3   # timed runs per configuration (first run is warmup)
MAXSTEPS = 2000

print(f"NGSolve {ngsolve.__version__}")
print()


def avg_ms(fn, nruns):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(nruns):
        fn()
    return (time.perf_counter() - t0) / nruns * 1000


# -----------------------------------------------------------------------
# 1. DevCGSolver — 3D Poisson, H1 order 2, Dirichlet on all faces
# -----------------------------------------------------------------------
print("=" * 78)
print("DevCGSolver — 3D Poisson, H1 order 2, Dirichlet all faces")
print(f"{'maxh':>5}  {'ndof':>8}  {'precond':>20}  {'iters':>6}  {'time(ms)':>9}  {'|sol|':>14}")
print("-" * 78)

for maxh in [0.3, 0.15, 0.08]:
    mesh = Mesh(unit_cube.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
    f = LinearForm(x*y*v*dx).Assemble()
    adev = a.mat.CreateDeviceMatrix()
    fdev = f.vec.CreateDeviceVector(copy=True)
    gfu  = GridFunction(fes)

    # point Jacobi
    pre_j  = a.mat.CreateSmoother(fes.FreeDofs())
    pdev_j = pre_j.CreateDeviceMatrix()
    cg_j   = ngscuda.DevCGSolver(mat=adev, pre=pdev_j,
                                  adev_raw=adev, cdev_raw=pdev_j,
                                  precision=1e-10, maxsteps=MAXSTEPS, printrates=False)
    t_j = avg_ms(lambda: cg_j.Mult(fdev, gfu.vec), NRUNS)
    print(f"{maxh:>5.2f}  {fes.ndof:>8d}  {'point-Jacobi':>20}  {cg_j.GetSteps():>6d}  {t_j:>9.2f}  {Norm(gfu.vec):>14.8e}")

    # block Gauss-Seidel
    blocks = fes.CreateSmoothingBlocks()
    pre_b  = a.mat.CreateBlockSmoother(blocks)
    pdev_b = pre_b.CreateDeviceMatrix()
    cg_b   = ngscuda.DevCGSolver(mat=adev, pre=pdev_b,
                                  adev_raw=adev, cdev_raw=pdev_b,
                                  precision=1e-10, maxsteps=MAXSTEPS, printrates=False)
    t_b = avg_ms(lambda: cg_b.Mult(fdev, gfu.vec), NRUNS)
    print(f"{maxh:>5.2f}  {fes.ndof:>8d}  {'block-Gauss-Seidel':>20}  {cg_b.GetSteps():>6d}  {t_b:>9.2f}  {Norm(gfu.vec):>14.8e}")
    print()

# -----------------------------------------------------------------------
# 2. DevTFQMRSolver — 2D convection-diffusion, H1 order 1, Dirichlet all edges
# -----------------------------------------------------------------------
print("=" * 78)
print("DevTFQMRSolver — 2D convection-diffusion, H1 order 1, Dirichlet all edges")
print(f"{'maxh':>5}  {'ndof':>8}  {'precond':>20}  {'iters':>6}  {'time(ms)':>9}  {'|sol|':>14}")
print("-" * 78)

for maxh in [0.1, 0.05, 0.03]:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=1, dirichlet=".*")
    u, v = fes.TnT()
    b_cf = CoefficientFunction((1, 0))
    a = BilinearForm(grad(u)*grad(v)*dx + b_cf*grad(u)*v*dx).Assemble()
    f = LinearForm(v*dx).Assemble()
    proj     = Projector(fes.FreeDofs(), True)
    adev     = a.mat.CreateDeviceMatrix()
    pdev_proj = proj.CreateDeviceMatrix()
    fdev     = f.vec.CreateDeviceVector(copy=True)
    gfu      = GridFunction(fes)

    # point Jacobi
    pre_j  = a.mat.CreateSmoother(fes.FreeDofs())
    pdev_j = pre_j.CreateDeviceMatrix()
    pa_j   = pdev_j @ adev
    rhs_j  = (pdev_j * fdev).Evaluate()
    tfq_j  = ngscuda.DevTFQMRSolver(mat=pre_j @ a.mat, pre=proj,
                                     adev_raw=pa_j, cdev_raw=pdev_proj,
                                     precision=1e-8, maxsteps=MAXSTEPS, printrates=False)
    t_j = avg_ms(lambda: tfq_j.Mult(rhs_j, gfu.vec), NRUNS)
    print(f"{maxh:>5.3f}  {fes.ndof:>8d}  {'point-Jacobi':>20}  {tfq_j.GetSteps():>6d}  {t_j:>9.2f}  {Norm(gfu.vec):>14.8e}")

    # block Gauss-Seidel
    blocks = fes.CreateSmoothingBlocks()
    pre_b  = a.mat.CreateBlockSmoother(blocks)
    pdev_b = pre_b.CreateDeviceMatrix()
    pa_b   = pdev_b @ adev
    rhs_b  = (pdev_b * fdev).Evaluate()
    tfq_b  = ngscuda.DevTFQMRSolver(mat=pre_b @ a.mat, pre=proj,
                                     adev_raw=pa_b, cdev_raw=pdev_proj,
                                     precision=1e-8, maxsteps=MAXSTEPS, printrates=False)
    t_b = avg_ms(lambda: tfq_b.Mult(rhs_b, gfu.vec), NRUNS)
    print(f"{maxh:>5.3f}  {fes.ndof:>8d}  {'block-Gauss-Seidel':>20}  {tfq_b.GetSteps():>6d}  {t_b:>9.2f}  {Norm(gfu.vec):>14.8e}")
    print()

print("Done.")
