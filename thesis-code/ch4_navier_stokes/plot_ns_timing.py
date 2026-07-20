"""
plot_ns_timing.py

Bar chart of NS velocity-solve timing from ns_timing-1535462.out.
Output: ns_timing_plot.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

labels  = ["CPU\n(CGSolver + BDDC)", "GPU no-graph\n(la.CGSolver)", "GPU WHILE graph\n(DevCGSolver)"]
means   = [202.0, 311.4, 236.9]
mins_   = [201.1, 310.6, 236.8]
maxs_   = [203.7, 312.6, 237.0]
errs_lo = [m - lo for m, lo in zip(means, mins_)]
errs_hi = [hi - m for m, hi in zip(means, maxs_)]

colors  = ["#5E6A7E", "#2563EB", "#D4601A"]

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#D0D4DC",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.7,
    "figure.dpi":        150,
})

fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(labels))
bars = ax.bar(x, means, width=0.5, color=colors, zorder=3,
              yerr=[errs_lo, errs_hi], capsize=4,
              error_kw=dict(elinewidth=1.2, ecolor="#3A4055", capthick=1.2))

# value labels above bars
for bar, val, lo, hi in zip(bars, means, mins_, maxs_):
    ax.text(bar.get_x() + bar.get_width() / 2,
            hi + 4,
            f"{val:.1f} ms",
            ha="center", va="bottom", fontsize=10, color="#3A4055")

# speedup annotations
ax.annotate("", xy=(2, means[2]), xytext=(1, means[1]),
            arrowprops=dict(arrowstyle="-", color="#9B2DC4",
                            linestyle="dashed", lw=1.2))
ax.text(1.5, (means[1] + means[2]) / 2 + 8,
        "1.31×  faster", ha="center", fontsize=9, color="#9B2DC4")

ax.set_ylabel("Velocity solve time per step (ms)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 370)
ax.set_title("NS velocity solve: CPU vs GPU (3-D Schäfer–Turek, MCS, ndof = 201 564, H100)",
             fontsize=9.5, pad=10, color="#3A4055")
ax.yaxis.set_minor_locator(plt.MultipleLocator(20))

fig.tight_layout()
fig.savefig("ns_timing_plot.png", dpi=150)
print("Saved ns_timing_plot.png")
