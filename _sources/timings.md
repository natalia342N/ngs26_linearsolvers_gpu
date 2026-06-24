# Results & Timings

All results on **H100 (Musica)**, 5 runs averaged, single GPU.

---

## Part 1: DevCGSolver — Poisson (symmetric SPD)

**Problem:** unit square, H1 order 2, varying mesh refinement, BlockSmoother preconditioner  
**Baseline:** Python `CGSolver` (tutorial 5.5.1 style) — loop on CPU, scalars round-trip through CPU  
**Comparison:** DevCGSolver no-graph (`UnifiedScalar`, C++ loop) and DevCGSolver WHILE graph (zero D2H)

| ndof | Python CG (ms) | No-graph (ms) | Conditional Graph (ms) | vs Python | vs No-graph |
|---:|---:|---:|---:|---:|---:|
| 501 | 5.9 | 3.0 | 1.7 | 3.40× | 1.70× |
| 997 | 8.5 | 4.3 | 2.4 | 3.53× | 1.77× |
| 1,961 | 11.7 | 5.9 | 3.5 | 3.30× | 1.68× |
| **5,277** | **19.1** | **9.7** | **5.2** | **3.67×** | **1.86×** |
| 11,825 | — | 13.9 | 7.9 | — | 1.76× |
| 46,741 | — | 29.8 | 18.0 | — | 1.65× |
| 185,809 | — | 90.6 | 65.8 | — | 1.38× |
| 514,637 | — | 183.9 | 155.3 | — | 1.18× |
| 1,157,621 | — | 338.3 | 307.1 | — | 1.10× |
| 4,624,201 | — | 1,306.2 | 1,275.6 | — | 1.02× |

**Peak speedup vs Python CG: 3.67× at ndof ≈ 5,000**  
**Peak speedup vs no-graph: 1.77× at ndof ≈ 5,000–12,000**

The speedup has two sources: (1) `UnifiedScalar` eliminates CPU round-trips for scalar quantities (α, β, ρ) by keeping all inner product results on GPU; (2) the CUDA conditional graph eliminates the per-iteration convergence check D2H sync and CPU launching overhead — the CPU calls the graph once and the entire loop runs GPU-resident. But the known limitation of large ndof is that computation dominates and both effects diminish.

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

