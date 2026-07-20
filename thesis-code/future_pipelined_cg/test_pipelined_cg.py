import os
import ngsolve
print(f"NGSolve: {ngsolve.__version__}")
from ngsolve import *
import ngsolve.ngscuda as ngscuda
from ngsolve import la
from netgen.occ import unit_cube
from time import perf_counter

NRUNS = 5

for maxh in [0.07, 0.05, 0.03]:
    mesh = Mesh(unit_cube.GenerateMesh(maxh=maxh))
    fes  = H1(mesh, order=2, dirichlet=".*")
    u, v = fes.TnT()
    dt = 0.01; nu = 0.001
    a = BilinearForm(fes)
    a += (1/dt)*u*v*dx
    a += nu*grad(u)*grad(v)*dx
    a.Assemble()
    blocks = fes.CreateSmoothingBlocks(blocktype="element")
    pre    = a.mat.CreateBlockSmoother(blocks)
    f      = LinearForm(1*v*dx).Assemble()
    gfu    = GridFunction(fes)
    fdev   = f.vec.CreateDeviceVector(copy=True)
    adev   = a.mat.CreateDeviceMatrix()
    predev = pre.CreateDeviceMatrix()
    ndof   = fes.ndof
    print(f"\n{'='*60}")
    print(f"ndof = {ndof}  (maxh={maxh})")

    gfu.vec.data = la.CGSolver(adev, predev, precision=1e-12, maxsteps=400) * fdev
    sol_ref = Norm(gfu.vec)
    print(f"ref |sol| = {sol_ref:.8e}")

    def run(env_set, env_unset, label):
        for k in env_unset:
            os.environ.pop(k, None)
        for k, v in env_set.items():
            os.environ[k] = v
        times = []
        for _ in range(NRUNS):
            solver = ngscuda.DevCGSolver(mat=adev, pre=predev, adev_raw=adev, cdev_raw=predev,
                                         precision=1e-12, maxsteps=400)
            t0 = perf_counter()
            gfu.vec.data = solver * fdev
            times.append(perf_counter() - t0)
        t = min(times)
        ok = 'OK' if abs(Norm(gfu.vec) - sol_ref) < 1e-4 else 'FAIL'
        return t, ok

    t_nogph,  ok = run({'NO_CUDA_GRAPH': '1'}, ['USE_DIAMOND_GRAPH'], 'no-graph')
    print(f"no-graph:        {t_nogph*1000:7.1f} ms  {ok}")

    t_while, ok = run({}, ['NO_CUDA_GRAPH','USE_DIAMOND_GRAPH'], 'WHILE')
    print(f"WHILE graph:     {t_while*1000:7.1f} ms  {ok}  vs no-graph: {t_nogph/t_while:.2f}x")

    t_diamond, ok = run({'USE_DIAMOND_GRAPH': '1'}, ['NO_CUDA_GRAPH'], 'diamond')
    print(f"diamond WHILE:   {t_diamond*1000:7.1f} ms  {ok}  vs no-graph: {t_nogph/t_diamond:.2f}x  vs WHILE: {t_while/t_diamond:.2f}x")
