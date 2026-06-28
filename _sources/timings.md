# Timings

All results on **H100 (Musica)**, 3 runs averaged, single GPU, `tol=1e-12`.

---

## Part 1: DevCGSolver — Poisson (symmetric SPD)

**Problem:** unit square, H1 order 2, Jacobi smoother preconditioner (`CreateSmoother`) — matches tutorial 5.5.1  
**Baselines:** CPU `CGSolver`, C++ `CGSolver` with device matrices (5.5.1 GPU baseline), DevCGSolver no-graph, DevCGSolver WHILE graph

| ndof | CPU CG (ms) | C++ dev (ms) | No-graph (ms) | WHILE graph (ms) | vs CPU | vs C++ dev | vs No-graph |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,961 | 195.8 | 6.8 | 5.7 | 3.5 | 56× | 1.95× | 1.64× |
| **5,277** | **276.9** | **10.8** | **9.2** | **5.0** | **55×** | **2.15×** | **1.83×** |
| **11,825** | **467.9** | **16.0** | **13.7** | **7.5** | **62×** | **2.13×** | **1.82×** |
| 46,741 | 926.5 | 36.0 | 29.5 | 16.8 | 55× | 2.15× | 1.76× |
| 95,225 | 1,540.5 | 51.9 | 44.6 | 26.4 | 58× | 1.96× | 1.69× |
| 185,809 | 5,179.1 | 80.5 | 73.9 | 48.8 | **106×** | 1.65× | 1.52× |
| 514,637 | 27,506.7 | 198.3 | 177.9 | 134.3 | **205×** | 1.48× | 1.32× |

**Peak speedup vs CPU: ~205× at ndof ≈ 514,000**  
**Peak speedup vs C++ dev (5.5.1 baseline): 2.15× at ndof ≈ 5,000–47,000**  
**Peak speedup vs no-graph: 1.83× at ndof ≈ 5,000–12,000**

"C++ dev" = `ngsolve.la.CGSolver` with device matrices — this is what tutorial 5.5.1 already gives you, with a C++ loop and D2H convergence check each iteration. "No-graph" = `DevCGSolver` with `NO_CUDA_GRAPH=1` — C++ with `UnifiedScalar`, no conditional graph.

> **Note:** CPU times are approximate and vary between nodes. GPU column times (C++ dev, no-graph, WHILE) are stable and consistent with tutorial reference results.

---

## Part 2: DevTFQMRSolver — 3D convection (non-symmetric)

**Problem:** unit cube, DG L2 order 2, convection-diffusion, block smoother preconditioner  
**Comparison:** Python `TFQMR` with device matrices (tutorial baseline), DevTFQMRSolver no-graph, WHILE graph

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

At ndof = 340,370, Python TFQMR (device) = 29.1 ms matches the tutorial reference result exactly.

The WHILE graph speedup depends on the problem — problems with cheaper iterations relative to sync overhead see larger speedup. On a simpler convection problem (H1 order 1, Jacobi preconditioner) the peak speedup reaches **2.05×** (see thesis).

---

## Summary

| | DevCGSolver (Poisson) | DevTFQMRSolver (convection DG) |
|---|---|---|
| vs CPU solver | up to **~205×** | — |
| vs 5.5.1 GPU baseline | up to **2.15×** | up to **2.7×** |
| vs no-graph | up to **1.83×** | up to **1.50×** |
| peak (vs no-graph) at ndof | ~5,000–12,000 | ~47,000 |
