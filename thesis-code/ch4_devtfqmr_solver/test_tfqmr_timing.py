from ngsolve import *
from netgen.occ import unit_square
import ngsolve.ngscuda as ngscuda
from ngsolve.krylovspace import TFQMR
import time, os

RUNS = 5

def run(maxh, pre_name, pre, pre1, a, f, fes):
    adev    = a.mat.CreateDeviceMatrix()
    predev  = pre.CreateDeviceMatrix()
    pre1dev = pre1.CreateDeviceMatrix()
    fdev    = f.vec.CreateDeviceVector(copy=True)
    pa_dev  = predev @ adev
    rhs_pre = (predev * fdev).Evaluate()

    # CPU reference
    gfu_ref = GridFunction(fes)
    gfu_ref.vec.data = TFQMR(mat=pre@a.mat, pre=pre1,
                              rhs=(pre*f.vec).Evaluate(),
                              maxsteps=800, printrates=False, tol=1e-8)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        gfu_ref.vec.data = TFQMR(mat=pre@a.mat, pre=pre1,
                                  rhs=(pre*f.vec).Evaluate(),
                                  maxsteps=800, printrates=False, tol=1e-8)
    cpu_t = (time.perf_counter() - t0) / RUNS

    def gpu_time(no_graph):
        if no_graph:
            os.environ["NO_CUDA_GRAPH"] = "1"
        gfu = GridFunction(fes)
        s = ngscuda.DevTFQMRSolver(mat=pre@a.mat, pre=pre1,
                                    adev_raw=pa_dev, cdev_raw=pre1dev,
                                    precision=1e-8, maxsteps=800, printrates=False)
        s.Mult(rhs_pre, gfu.vec)          # warmup
        t0 = time.perf_counter()
        for _ in range(RUNS):
            gfu.vec[:] = 0
            s.Mult(rhs_pre, gfu.vec)
        t = (time.perf_counter() - t0) / RUNS
        if no_graph:
            del os.environ["NO_CUDA_GRAPH"]
        diff = Norm(gfu_ref.vec - gfu.vec)
        return t, diff

    t_ng, e_ng = gpu_time(no_graph=True)
    t_g,  e_g  = gpu_time(no_graph=False)

    print(f"  ndof={fes.ndof:6d}  CPU={cpu_t*1e3:8.1f}ms  "
          f"no-graph={t_ng*1e3:8.1f}ms  graph={t_g*1e3:8.1f}ms  "
          f"speedup={t_ng/t_g:.2f}x  err_ng={e_ng:.1e}  err_g={e_g:.1e}")

for maxh in [0.05, 0.02, 0.01, 0.005]:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=1, dirichlet=".*")
    u, v = fes.TnT()
    wind = CF((1, 0))
    a    = BilinearForm((grad(u)*grad(v) + wind*grad(u)*v)*dx).Assemble()
    f    = LinearForm(1*v*dx).Assemble()

    pre_jac   = a.mat.CreateSmoother(fes.FreeDofs())
    blocks    = fes.CreateSmoothingBlocks()
    pre_block = a.mat.CreateBlockSmoother(blocks)
    pre1      = Projector(fes.FreeDofs(), True)

    if maxh == 0.05:
        print("=== Jacobi preconditioner ===")
        print(f"  {'ndof':>6}  {'CPU':>10}  {'no-graph':>12}  {'graph':>10}  {'speedup':>8}  {'err_ng':>8}  {'err_g':>8}")
        print("  " + "-"*85)
    run(maxh, "jacobi", pre_jac, pre1, a, f, fes)

print()
for maxh in [0.05, 0.02, 0.01, 0.005]:
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=1, dirichlet=".*")
    u, v = fes.TnT()
    wind = CF((1, 0))
    a    = BilinearForm((grad(u)*grad(v) + wind*grad(u)*v)*dx).Assemble()
    f    = LinearForm(1*v*dx).Assemble()

    blocks    = fes.CreateSmoothingBlocks()
    pre_block = a.mat.CreateBlockSmoother(blocks)
    pre1      = Projector(fes.FreeDofs(), True)

    if maxh == 0.05:
        print("=== Block Jacobi preconditioner ===")
        print(f"  {'ndof':>6}  {'CPU':>10}  {'no-graph':>12}  {'graph':>10}  {'speedup':>8}  {'err_ng':>8}  {'err_g':>8}")
        print("  " + "-"*85)
    run(maxh, "block", pre_block, pre1, a, f, fes)
