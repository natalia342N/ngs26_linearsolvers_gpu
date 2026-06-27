# Iterative Linear Solvers on GPUs

**Natalia Tylek — TU Wien**

**NGSolve User Meeting 2026, Winterthur**

---

**This presentation:** two model problems demonstrating the use of CUDA Graph solvers

| | Problem | Solver | Status |
|---|---|---|---|
| Part 1 | Poisson (symmetric SPD) | `DevCGSolver` | available in pre-release |
| Part 2 | Convection (non-symmetric) | `DevTFQMRSolver` | not yet in pre-release |

---

**Thesis context:** both solvers are needed in the IPCS Navier–Stokes timestepper

| NS component | Symmetric | Solver | GPU solver |
|---|---|---|---|
| Convection | No | TFQMR | `DevTFQMRSolver` |
| Viscous / mass | Yes (SPD) | CG + preconditioner | `DevCGSolver` |
| Pressure proj. | Yes (SPD) | CG + preconditioner | `DevCGSolver` |

*GPU Implementations and CUDA Graph Acceleration of Krylov Solvers for Incompressible Navier–Stokes in NGSolve*

---

## How to use

- **Google Colab** — no installation, runs in the browser  
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/natalia342N/ngs26_linearsolvers_gpu/blob/main/docs/ngsolve_meeting_tutorial.ipynb)
- **Local GPU** — `pip install ngsolve`, then open the tutorial notebook
- **HPC cluster** — see [Getting Started](installation.md) for build and job submission
