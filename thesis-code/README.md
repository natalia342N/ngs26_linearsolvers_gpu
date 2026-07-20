# Thesis Code — GPU-Resident Krylov Solvers in NGSolve

Natalia Tylek, TU Wien, 2026.
Scripts and benchmarks for the master's thesis on CUDA graph-based Krylov solvers in NGSolve (`ngscuda`).

## Structure

| Folder | Thesis chapter | Contents |
|---|---|---|
| `ch3_infrastructure/` | Ch. 3 | CUDA graph building blocks, UnifiedScalar tests |
| `ch3_breakeven/` | Ch. 3 | Kernel-launch breakeven analysis (`.cu`, `.py`, results) |
| `ch4_devcg_solver/` | Ch. 4.1 | DevCGSolver — Poisson/CG scaling benchmarks |
| `ch4_devtfqmr_solver/` | Ch. 4.2 | DevTFQMRSolver — convection/TFQMR benchmarks |
| `ch4_navier_stokes/` | Ch. 4.3 | Navier-Stokes timing + 3D webgui visualization |
| `ch4_profiling/` | Ch. 4 | Nsight Systems profiling scripts |
| `future_diamond_graph/` | Ch. 5 (limitations) | Diamond-graph negative result |
| `future_lobpcg/` | Future work | GPU LOBPCG experiment |
| `future_pipelined_cg/` | Future work | Pipelined CG prototype |

Each folder has a `slurm/` subfolder with the corresponding SBATCH submit scripts.

## Dependencies

- NGSolve built from source with `ngscuda` enabled
- CUDA 12.9, GCC 14.2, OpenCASCADE 7.9.1
- MUSICA cluster: partition `zen4_0768_h100x4` (NVIDIA H100)
