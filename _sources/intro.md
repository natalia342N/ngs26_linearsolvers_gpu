# Iterative Linear Solvers on GPUs

**Natalia Tylek — TU Wien**

**NGSolve User Meeting 2026, Winterthur**

---

```{note}
**This tutorial documents the CUDA-specific implementation as of June 2026.**

Since then, the work has been folded into NGSolve's general GPU layer.
The `ngsolve.gpu` module now dispatches across backends — CUDA via
`ngscuda`, Apple GPUs via `ngsmetal`, and a host reference device
otherwise — and the graph-recording idea developed here has moved into
that shared framework. The API may differ from what you see below.

For the current interface, see the official tutorial:
[Solving the Poisson Equation on devices](https://docu.ngsolve.org/latest/i-tutorials/unit-5.5-cuda/poisson_cuda.html)

The material here remains useful for the reasoning behind the design:
why per-iteration CPU synchronisation dominates GPU Krylov solvers, and
how capturing the iteration loop removes it.
```


**This presentation:** two model problems demonstrating the use of CUDA Graph solvers

| | Problem | Solver | Status |
|---|---|---|---|
| Part 1 | Poisson (symmetric SPD) | `DevCGSolver` | available in pre-release |
| Part 2 | Convection (non-symmetric) | `DevTFQMRSolver` | (not) available in pre-release |

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
