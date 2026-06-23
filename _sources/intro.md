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