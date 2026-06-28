# Timings

All results on **H100 (Musica)**, 3 runs averaged, single GPU.
**Software:** NGSolve 6.2.2604-9-gf15d395df, CUDA 12.9.  
**Hardware:** NVIDIA H100 (GPU) | AMD EPYC Zen4, dual-socket, 384 logical threads. 

---

## Part 1: DevCGSolver — Poisson (symmetric SPD)

**Problem:** unit square, H1 order 2, Jacobi smoother preconditioner (`CreateSmoother`) — matches tutorial 5.5.1  
**Baselines:** CPU `CGSolver`, C++ `CGSolver` with device matrices, DevCGSolver no-graph, DevCGSolver Conditional While Graph

| ndof | CPU all-T (ms) | CPU 1T (ms) | C++ dev (ms) | No-graph (ms) | Conditional While Graph (ms) | vs 1T | vs C++ dev | vs No-graph |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,961 | 98.5 | 1.9 | 6.8 | 5.7 | 3.5 | 0.5× | 1.95× | 1.64× |
| **5,277** | **346.6** | **8.5** | **10.8** | **9.1** | **5.0** | **1.7×** | **2.14×** | **1.81×** |
| **11,825** | **308.5** | **29.0** | **16.0** | **13.7** | **7.5** | **3.9×** | **2.12×** | **1.81×** |
| 46,741 | 1,351.5 | 235.9 | 35.2 | 29.2 | 16.8 | 14.0× | 2.09× | 1.74× |
| 95,225 | 2,467.0 | 695.0 | 51.6 | 44.6 | 27.1 | 25.7× | 1.91× | 1.65× |
| 185,809 | 4,607.0 | 1,973.7 | 81.4 | 74.9 | 49.5 | 39.9× | 1.64× | 1.51× |
| 514,637 | 28,133.6 | 9,783.5 | 198.0 | 177.9 | 134.6 | 72.7× | 1.47× | 1.32× |

> - **CPU all-T**: NGSolve TaskManager with all 384 logical threads on the node (dual-socket AMD EPYC Zen4, ignores SLURM `--cpus-per-task=22`). 
> - **CPU 1T**: no TaskManager, single-threaded reference — unambiguous lower bound for CPU performance.  
> GPU column times (C++ dev, no-graph, Conditional While Graph) are stable and node-independent.

---

## Part 2: DevTFQMRSolver — 3D convection (non-symmetric)

**Problem:** unit cube, DG L2 order 2, convection-diffusion, block smoother preconditioner  
**Comparison:** Python `TFQMR` with device matrices, DevTFQMRSolver no-graph, DevTFQMRSolver Conditional While Graph

| ndof | Python TFQMR (ms) | No-graph (ms) | Conditional While Graph (ms) | vs Python dev | vs No-graph |
|---:|---:|---:|---:|---:|---:|
| 2,060 | 3.6 | 1.8 | 1.5 | 2.4× | 1.26× |
| 6,520 | 4.6 | 2.4 | 1.7 | 2.7× | 1.39× |
| 20,250 | 6.3 | 3.1 | 2.3 | 2.7× | 1.38× |
| **47,810** | **8.5** | **4.7** | **3.2** | **2.7×** | **1.50×** |
| 149,650 | 15.5 | 9.9 | 8.0 | 1.9× | 1.23× |
| 340,370 | 29.1 | 21.3 | 19.1 | 1.5× | 1.12× |

