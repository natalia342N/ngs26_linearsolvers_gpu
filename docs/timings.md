#  Timings

All results on **H100 (Musica)**, 5 runs averaged, single GPU.

---

## Part 1: DevCGSolver — Poisson (symmetric SPD)

**Problem:** unit square, H1 order 2, varying mesh refinement  
**Comparison:** DevCGSolver (WHILE graph) vs DevCGSolver (no graph, `NO_CUDA_GRAPH=1`)

| ndof | No-graph (ms) | WHILE graph (ms) | Speedup |
|---:|---:|---:|---:|
| 501 | 2.9 | 2.1 | 1.38× |
| 997 | 4.1 | 2.4 | 1.68× |
| 1,961 | 5.6 | 3.2 | 1.74× |
| **5,277** | **9.2** | **5.2** | **1.77×** |
| 11,825 | 13.9 | 7.9 | 1.76× |
| 46,741 | 29.8 | 18.0 | 1.65× |
| 185,809 | 90.6 | 65.8 | 1.38× |
| 514,637 | 183.9 | 155.3 | 1.18× |
| 1,157,621 | 338.3 | 307.1 | 1.10× |
| 4,624,201 | 1,306.2 | 1,275.6 | 1.02× |

**Peak speedup: 1.77× at ndof ≈ 5,000–12,000**

The speedup is largest at small–medium problem sizes where GPU compute per iteration
is fast and the per-iteration D2H synchronization overhead dominates. At large ndof
the computation dominates and the two approaches converge.

---

## Part 2: DevTFQMRSolver — 3D convection (non-symmetric)

**Problem:** unit cube, DG L2 order 2, ndof = 340,370  
**Comparison:** DevTFQMRSolver (WHILE graph) vs CPU and GPU baselines

| Solver | Time (ms) | Speedup vs CPU |
|---|---:|---:|
| CPU TFQMR | 466.9 | 1× |
| GPU Python TFQMR | 21.8 | 21× |
| **DevTFQMRSolver (WHILE graph)** | **14.4** | **32×** |

**DevTFQMR vs GPU Python TFQMR: 1.51× at ndof = 340,370**

The WHILE graph speedup over the Python TFQMR baseline is larger at smaller problem
sizes (more iterations relative to compute per iteration). The 32× speedup over CPU
demonstrates the GPU acceleration independent of the graph approach.
