"""
plot_breakeven.py

Reads CSV output from benchmark_breakeven and generates the break-even plot.

Usage:
    python3 plot_breakeven.py breakeven_data.csv
    ./benchmark_breakeven | python3 plot_breakeven.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def parse_csv(lines):
    results = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) < 6:
            continue
        try:
            N = int(parts[0])
            results[N] = dict(
                t_chain     = float(parts[1]),
                t_per_kernel= float(parts[2]),
                t_inst      = float(parts[3]),
                t_launch    = float(parts[4]),
                r_break     = float(parts[5]),
            )
        except ValueError:
            continue
    return results


def plot(results):
    PALETTE   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    HIGHLIGHT = {1: 'N=1', 17: 'N=17'}
    R_MAX     = 25   # zoom in — all crossovers happen before R=25
    R         = np.linspace(0, R_MAX, 600)

    fig, ax = plt.subplots(figsize=(8, 5))
    series_handles = []

    # Y ceiling: ignore the N=1 graph line's tall start — clip at 600 us
    Y_MAX = 600

    for i, (N, d) in enumerate(sorted(results.items())):
        col = PALETTE[i % len(PALETTE)]
        lw  = 2.2 if N in HIGHLIGHT else 1.4

        y_nogr = R * d['t_chain']
        y_gr   = d['t_inst'] + R * d['t_launch']

        ax.plot(R, y_nogr, color=col, ls='--', lw=lw)
        ax.plot(R, y_gr,   color=col, ls='-',  lw=lw)

        if 0 < d['r_break'] < R_MAX:
            y_cross = d['t_inst'] + d['r_break'] * d['t_launch']
            ax.plot(d['r_break'], y_cross, 'o', color=col, ms=6, zorder=5)
            ax.annotate(f"R≈{d['r_break']:.0f}",
                        xy=(d['r_break'], y_cross),
                        xytext=(d['r_break'] + 0.4, y_cross + 18),
                        fontsize=7.5, color=col)

        label = HIGHLIGHT.get(N, f'N={N}')
        series_handles.append(mlines.Line2D([], [], color=col, lw=lw, label=label))

    dash_line  = mlines.Line2D([], [], color='k', ls='--', lw=1.4, label='no-graph (individual launches)')
    solid_line = mlines.Line2D([], [], color='k', ls='-',  lw=1.4, label='graph (instantiation + launches)')
    ax.legend(handles=series_handles + [dash_line, solid_line],
              fontsize=8, loc='upper left', framealpha=0.95, ncol=1)

    ax.set_xlabel('Solver iterations $R$', fontsize=11)
    ax.set_ylabel('Cumulative kernel launch overhead ($\\mu$s)', fontsize=11)
    ax.set_title('CUDA graph break-even on H100: launch overhead vs. iterations', fontsize=11)
    ax.set_xlim(0, R_MAX)
    ax.set_ylim(0, Y_MAX)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('breakeven_plot.pdf', bbox_inches='tight')
    plt.savefig('breakeven_plot.png', dpi=150, bbox_inches='tight')
    print('Saved breakeven_plot.pdf / .png')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    results = parse_csv(lines)
    if not results:
        print('No data. Usage: python3 plot_breakeven.py breakeven_data.csv')
        sys.exit(1)

    print(f'Loaded N = {sorted(results.keys())}')
    for N, d in sorted(results.items()):
        print(f'  N={N:2d}: chain={d["t_chain"]:.2f} us  '
              f'inst={d["t_inst"]:.2f} us  '
              f'launch={d["t_launch"]:.2f} us  '
              f'R_break={d["r_break"]:.2f}')
    plot(results)
