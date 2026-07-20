"""
plot_convergence.py

Generate convergence history plot for CG comparison:
  - DevCGSolver, no graph
  - DevCGSolver, WHILE graph
  - Python CG (device matrices)

Data from convergence-1535523.out.
Run: python plot_convergence.py
Output: convergence_plot.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── data ──────────────────────────────────────────────────────────────────────

RES_CXX = np.array([
    4.201391e-2, 5.528681e-2, 7.134289e-2, 5.325693e-2, 4.128887e-2,
    3.005530e-2, 3.840821e-2, 3.042671e-2, 2.505406e-2, 2.050650e-2,
    1.567803e-2, 1.472923e-2, 1.043897e-2, 7.338887e-3, 4.720338e-3,
    2.989260e-3, 2.033112e-3, 1.348514e-3, 8.996263e-4, 6.173866e-4,
    4.551657e-4, 3.927770e-4, 2.970988e-4, 2.219576e-4, 1.725809e-4,
    1.310712e-4, 1.081483e-4, 1.027188e-4, 1.011187e-4, 8.191801e-5,
    5.762549e-5, 4.618446e-5, 3.325132e-5, 2.476052e-5, 2.229843e-5,
    1.846182e-5, 1.569235e-5, 1.498358e-5, 1.422497e-5, 1.094439e-5,
    7.374892e-6, 5.060108e-6, 3.376738e-6, 2.370884e-6, 2.003515e-6,
    1.927897e-6, 1.561738e-6, 1.069760e-6, 7.383092e-7, 5.534322e-7,
    3.863051e-7, 2.916200e-7, 2.340831e-7, 1.778043e-7, 1.316795e-7,
    9.395596e-8, 6.901425e-8, 5.140977e-8, 3.791081e-8, 2.991547e-8,
    2.415032e-8, 1.809704e-8, 1.282545e-8, 8.562422e-9, 5.576723e-9,
    3.741769e-9, 2.504207e-9, 1.820344e-9, 1.346311e-9, 9.316423e-10,
    6.351790e-10, 4.827900e-10, 3.907071e-10, 3.593930e-10, 3.560650e-10,
    3.431926e-10, 2.899819e-10, 2.320019e-10,
])

# Python CG records r₀ before the first step, then the same values as CXX
RES_PYTHON = np.concatenate([[2.670909e-2], RES_CXX])

TOL = 1e-8

iters_cxx    = np.arange(1, len(RES_CXX) + 1)          # 1 … 78
iters_python = np.arange(0, len(RES_PYTHON))            # 0 … 78

# ── style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#D0D4DC",
    "grid.linestyle":     "--",
    "grid.linewidth":     0.6,
    "grid.alpha":         0.7,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.minor.size":   2,
    "ytick.minor.size":   2,
    "figure.dpi":         150,
})

C_NOGRAPH = "#2563EB"   # blue
C_GRAPH   = "#D4601A"   # warm orange
C_PYTHON  = "#16924A"   # green
C_TOL     = "#9B2DC4"   # purple

# ── figure ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

EVERY = 8   # marker spacing (iterations)

ax.semilogy(iters_cxx, RES_CXX,
            color=C_NOGRAPH, linewidth=2.0,
            marker="o", markersize=6, markevery=EVERY,
            markerfacecolor="white", markeredgewidth=1.8,
            label="DevCGSolver — no graph")

ax.semilogy(iters_cxx, RES_CXX,
            color=C_GRAPH, linewidth=1.8, linestyle=(0, (6, 4)),
            marker="s", markersize=6, markevery=(4, EVERY),
            markerfacecolor="white", markeredgewidth=1.8,
            label="DevCGSolver — WHILE graph (overlaps exactly)")

ax.semilogy(iters_python, RES_PYTHON,
            color=C_PYTHON, linewidth=1.8,
            marker="^", markersize=6, markevery=(2, EVERY),
            markerfacecolor="white", markeredgewidth=1.8,
            label="Python CG (device matrices)")

ax.axhline(TOL, color=C_TOL, linewidth=1.2, linestyle=":",
           alpha=0.75, label="Tolerance $10^{-8}$")

# axes
ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel(r"Residual norm $\|r_k\|$", fontsize=11)
ax.set_xlim(0, 80)
ax.set_ylim(5e-11, 0.3)
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))

# legend
leg = ax.legend(fontsize=9.5, framealpha=0.92, edgecolor="#CBD0DA",
                loc="upper right")

# title
ax.set_title("CG convergence history — 2D Poisson, H1 order 2, Jacobi preconditioner, H100",
             fontsize=10, pad=10, color="#3A4055")

fig.tight_layout()
fig.savefig("convergence_plot.png", dpi=150)
print("Saved convergence_plot.png")
