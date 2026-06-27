# Timings

All results on **H100 (Musica)**, 5 runs averaged, single GPU, `tol=1e-12`.

---

## Part 1: DevCGSolver — Poisson (symmetric SPD)

**Problem:** unit square, H1 order 2, Jacobi smoother preconditioner (`CreateSmoother`) — matches tutorial 5.5.1  
**Baselines:** CPU `CGSolver`, Python `CGSolver` with device matrices (naive GPU), DevCGSolver no-graph, DevCGSolver WHILE graph

| ndof | CPU CG (ms) | Python dev (ms) | No-graph (ms) | WHILE graph (ms) | vs CPU | vs Python dev | vs No-graph |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 501 | 146.8 | 5.7 | 3.2 | 1.9 | 78× | 3.0× | 1.74× |
| 997 | 234.7 | 7.5 | 4.2 | 2.4 | 99× | 3.2× | 1.79× |
| 1,961 | 232.4 | 10.9 | 5.8 | 3.5 | 67× | 3.1× | 1.67× |
| **5,277** | **389.1** | **22.0** | **9.3** | **5.0** | **77×** | **4.4×** | **1.85×** |
| **11,825** | **657.3** | **43.5** | **13.9** | **7.5** | **88×** | **5.8×** | **1.85×** |
| 46,741 | 1,342.8 | 182.2 | 29.7 | 16.8 | 80× | 10.8× | 1.76× |
| 95,225 | 2,603.6 | 687.9 | 45.1 | 26.6 | 98× | 25.9× | 1.70× |
| 185,809 | 5,824.4 | 1,634.0 | 72.7 | 47.0 | **124×** | 34.7× | 1.55× |
| 514,637 | 30,193.3 | 5,170.8 | 179.7 | 135.5 | 223× | **38.2×** | 1.33× |

**Peak speedup vs CPU: ~124× at ndof ≈ 185,000**  
**Peak speedup vs Python dev (device matrices): ~38× at ndof ≈ 514,000**  
**Peak speedup vs no-graph: 1.85× at ndof ≈ 5,000–12,000**

Each step adds an improvement:
- **CPU → Python dev**: device matrices alone give ~3–38× (GPU compute, but Python loop overhead per iteration limits scalability)
- **Python dev → No-graph**: C++ implementation with `UnifiedScalar` eliminates Python per-iteration overhead — another 2–5× at moderate ndof, growing to 28× at large ndof
- **No-graph → WHILE graph**: CUDA conditional graph eliminates the convergence check sync — up to **1.85×** more

> **Note:** "Python dev" = Python `CGSolver` with device matrices — GPU compute but scalars return to CPU each iteration. "No-graph" = `DevCGSolver` with `NO_CUDA_GRAPH=1` — C++ with `UnifiedScalar`, no conditional graph.

---

## Part 2: DevTFQMRSolver — 3D convection (non-symmetric)

**Problem:** unit cube, DG L2 order 2, convection-diffusion, block smoother preconditioner  
**Comparison:** Python `TFQMR` with device matrices, DevTFQMRSolver no-graph, WHILE graph

| ndof | Python TFQMR (ms) | No-graph (ms) | WHILE graph (ms) | vs Python dev | vs No-graph |
|---:|---:|---:|---:|---:|---:|
| 2,060 | 3.6 | 1.8 | 1.5 | 2.4× | 1.26× |
| 6,520 | 4.6 | 2.4 | 1.7 | 2.7× | 1.39× |
| 20,250 | 6.3 | 3.1 | 2.3 | 2.7× | 1.38× |
| **47,810** | **8.5** | **4.7** | **3.2** | **2.7×** | **1.50×** |
| 149,650 | 15.5 | 9.9 | 8.0 | 1.9× | 1.23× |
| 340,370 | 29.1 | 21.3 | 19.1 | 1.5× | 1.12× |

**Peak speedup vs Python dev: ~2.7× at ndof ≈ 6,000–48,000**  
**Peak speedup vs no-graph: 1.50× at ndof ≈ 47,000**

At ndof = 340,370, the Python TFQMR (device matrices) time of 29.1 ms matches the tutorial reference result exactly.

The WHILE graph speedup depends on the problem setup — problems with cheaper iterations
(less work per iteration relative to the sync overhead) see larger speedup. For example,
on a simpler convection problem (H1 order 1, Jacobi preconditioner) the peak speedup
reaches **2.05×** (see thesis).

---

## Summary

| | DevCGSolver (Poisson) | DevTFQMRSolver (convection DG) |
|---|---|---|
| vs CPU Python solver | up to **~124×** | — |
| vs Python dev (device matrices) | up to **~38×** | up to **~2.7×** |
| vs no-graph | up to **1.85×** | up to **1.50×** |
| peak (vs no-graph) at ndof | ~5,000–12,000 | ~47,000 |
