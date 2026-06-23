# Linear Solvers on GPUs

**Natalia Tylek — TU Wien**

**NGSolve User Meeting 2026, June 29 – July 1, Winterthur**

---

**This presentation:** two model problems demonstrating the CUDA WHILE graph solvers

| | Problem | Solver |
|---|---|---|
| Part 1 | Poisson (symmetric SPD) | `DevCGSolver` | implemented, available publicly |
| Part 2 | 3D convection (non-symmetric) | `DevTFQMRSolver` | implemented, not yet available publicly |

---

**Thesis context:** both solvers are needed in the IPCS Navier–Stokes timestepper

| NS component | Symmetric | Constant | Solver | GPU solver |
|---|---|---|---|---|
| Convection | No | No | TFQMR | `DevTFQMRSolver`  |
| Viscous / mass | Yes (SPD) | Yes | CG + BDDC | `DevCGSolver` |
| Pressure proj. | Yes (SPD) | Yes | CG + H1AMG | `DevCGSolver` |
| Full NS timestep | | | all 3 combined |  to be integrated |


*GPU Implementations and CUDA Graph Acceleration of Krylov Solvers for Incompressible Navier–Stokes in NGSolve*


---

## How to use

- **Google Colab** — no installation, runs in the browser  
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/natalia342N/ngs26_linearsolvers_gpu/blob/main/docs/ngsolve_meeting_tutorial.ipynb)
- **Local GPU** — `pip install ngsolve`, then open the tutorial notebook
- **HPC cluster** — see [Getting Started](installation.md) for build and job submission
