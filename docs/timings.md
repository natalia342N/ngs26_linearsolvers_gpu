# Timings

All results on **H100 (Musica)**, 5 runs averaged, single GPU.

---

## Part 1: DevCGSolver — Poisson (symmetric SPD)

**Problem:** unit square, H1 order 2, Jacobi smoother preconditioner (`CreateSmoother`) — matches tutorial 5.5.1  
**Comparison:** CPU `CGSolver` (tutorial 5.5.1 baseline), DevCGSolver no-graph, DevCGSolver WHILE graph

| ndof | CPU CG (ms) | No-graph (ms) | WHILE graph (ms) | vs CPU | vs No-graph |
|---:|---:|---:|---:|---:|---:|
| 501 | 132.7 | 2.5 | 1.5 | 89× | 1.70× |
| 997 | 106.8 | 3.3 | 1.9 | 57× | 1.77× |
| 1,961 | 143.0 | 4.5 | 2.9 | 50× | 1.57× |
| **5,277** | **342.5** | **7.2** | **4.0** | **86×** | **1.81×** |

**Peak speedup vs CPU: ~90× at ndof ≈ 500**  
**Peak speedup vs no-graph: 1.81× at ndof ≈ 5,000**

The speedup has two sources: (1) `UnifiedScalar` eliminates CPU round-trips for scalar quantities (α, β, ρ) by keeping all inner product results on GPU; (2) the CUDA conditional graph eliminates the per-iteration convergence check D2H sync and CPU dispatch overhead — the CPU launches once and the entire loop runs GPU-resident. At large ndof computation dominates and both effects diminish.

> **Note:** CPU CG times are high for small ndof due to Python loop overhead per iteration. The vs-CPU speedup reflects the combined benefit of GPU acceleration, `UnifiedScalar`, and the CUDA conditional graph.

---

## Part 2: DevTFQMRSolver — 3D convection (non-symmetric)

**Problem:** unit cube, DG L2 order 2, convection-diffusion, block smoother preconditioner  
**Comparison:** DevTFQMRSolver no-graph vs WHILE graph, varying mesh size

| ndof | No-graph (ms) | WHILE graph (ms) | Speedup |
|---:|---:|---:|---:|
| 2,060 | 1.5 | 1.2 | 1.17× |
| 6,520 | 1.9 | 1.5 | 1.33× |
| 20,250 | 2.4 | 1.9 | 1.27× |
| **47,810** | **4.2** | **2.8** | **1.54×** |
| 149,650 | 7.5 | 6.7 | 1.13× |
| 340,370 | 16.1 | 14.6 | 1.10× |

**Peak speedup: 1.54× at ndof ≈ 47,000**

At ndof = 340,370, comparison against other baselines:

| Solver | Time (ms) | vs CPU |
|---|---:|---:|
| CPU TFQMR | 466.9 | 1× |
| GPU Python TFQMR | 21.8 | 21× |
| DevTFQMR no-graph | 16.1 | 29× |
| **DevTFQMR WHILE graph** | **14.4** | **32×** |

The WHILE graph speedup depends on the problem setup — problems with cheaper iterations
(less work per iteration relative to the sync overhead) see larger speedup. For example,
on a simpler convection problem (H1 order 1, Jacobi preconditioner) the peak speedup
reaches **2.05×** (see thesis).

---

## Summary

| | DevCGSolver (Poisson) | DevTFQMRSolver (convection DG) |
|---|---|---|
| vs CPU solver | up to **~90×** | up to **32×** |
| vs no-graph | up to **1.81×** | up to **1.54×** |
| peak at ndof | ~5,000 | ~47,000 |
