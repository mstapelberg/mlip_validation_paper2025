import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from matplotlib.lines import Line2D

# ── Style: clean white academic ────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri', 'DejaVu Sans', 'Arial'],
    'axes.facecolor': '#FFFFFF',
    'figure.facecolor': '#FFFFFF',
    'text.color': '#1A1A2E',
    'axes.labelcolor': '#333333',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
    'axes.edgecolor': '#CCCCCC',
    'grid.color': '#E8E8E8',
    'grid.alpha': 0.8,
})

# ── Colors ─────────────────────────────────────────────────────────────────
DFT_COLOR = '#4477AA'   # steel blue
MLIP_COLOR = '#228833'  # forest green
THIS_COLOR = '#EE6677'  # soft red / coral
ACCENT_GRAY = '#888888'

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
DETAILED_OUTPUT_STEM = 'compute_frontier_white'
SIMPLE_OUTPUT_STEM = 'compute_frontier_white_simple'
DEFAULT_Y_LABEL = 'System size (atoms) at DFT or near-DFT fidelity'
SIMPLE_Y_LABEL = 'System size (atoms)\nat DFT or near-DFT fidelity'

# ── Data ───────────────────────────────────────────────────────────────────
dft_years = [1999, 2009, 2014, 2020, 2023, 2023.15]
dft_atoms = [300, 32768, 5898, 512, 300, 3584]
dft_marker = ['routine', 'advanced', 'heroic', 'software', 'routine', 'heroic']
dft_env_y = [1999, 2014, 2023.15]
dft_env_a = [300, 5898, 3584]

mlip_years = [2019, 2021, 2023, 2023.15, 2025, 2025.15, 2026]
mlip_atoms = [16000, 113246208, 100640512, 8100000, 27648000, 1024000000, 612000]
mlip_env_y = [2019, 2021, 2023, 2025.15]
mlip_env_a = [16000, 113246208, 100640512, 1024000000]


def add_manual_label(
    ax,
    point_xy,
    text,
    text_xy,
    *,
    color,
    ha='left',
    va='center',
    fontsize=6,
    fontweight=None,
    line_color=None,
    arrowstyle='-',
    line_width=0.6,
    line_alpha=0.4,
    bbox=None,
    zorder=8,
):
    # Use data coordinates so label placement stays intuitive on the log y-axis.
    ax.annotate(
        text,
        xy=point_xy,
        xytext=text_xy,
        textcoords='data',
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        ha=ha,
        va=va,
        zorder=zorder,
        annotation_clip=False,
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=line_color or color,
            lw=line_width,
            alpha=line_alpha,
            shrinkA=4,
            shrinkB=4,
        ),
        bbox=bbox,
    )


def plot_dft_points(ax, *, scale=1.0):
    for yr, at, mt in zip(dft_years, dft_atoms, dft_marker):
        if mt == 'routine':
            ax.plot(
                yr,
                at,
                'o',
                color=DFT_COLOR,
                markersize=6 * scale,
                zorder=5,
                markeredgecolor=DFT_COLOR,
                markeredgewidth=1,
            )
        elif mt == 'advanced':
            ax.plot(
                yr,
                at,
                'o',
                color='white',
                markersize=7 * scale,
                zorder=5,
                markeredgecolor=DFT_COLOR,
                markeredgewidth=1.5,
            )
        elif mt == 'heroic':
            ax.plot(
                yr,
                at,
                '*',
                color=DFT_COLOR,
                markersize=11 * scale,
                zorder=5,
                markeredgecolor=DFT_COLOR,
                markeredgewidth=0.5,
            )
        elif mt == 'software':
            ax.plot(yr, at, 's', color=DFT_COLOR, markersize=5 * scale, zorder=5, alpha=0.5)


def plot_mlip_points(ax, *, scale=1.0):
    for i, (yr, at) in enumerate(zip(mlip_years, mlip_atoms)):
        if i == len(mlip_years) - 1:
            ax.plot(
                yr,
                at,
                'D',
                color=THIS_COLOR,
                markersize=10 * scale,
                zorder=10,
                markeredgecolor='#CC3355',
                markeredgewidth=1.5,
            )
        else:
            ax.plot(
                yr,
                at,
                'o',
                color=MLIP_COLOR,
                markersize=7 * scale,
                zorder=5,
                markeredgecolor=MLIP_COLOR,
                markeredgewidth=1,
            )


def style_axes(
    ax,
    *,
    y_label,
    xlim,
    x_label_size,
    y_label_size,
    tick_label_size,
    label_weight='normal',
    axis_color='#CCCCCC',
    tick_color='#555555',
    show_grid=True,
    show_minor_x=True,
    show_minor_y=True,
):
    ax.set_yscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(15, 5e10)
    ax.set_xlabel('Year', fontsize=x_label_size, labelpad=10, fontweight=label_weight)
    ax.set_ylabel(y_label, fontsize=y_label_size, labelpad=12, fontweight=label_weight)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1) if show_minor_x else ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=12))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    if not show_minor_y:
        ax.yaxis.set_minor_locator(ticker.NullLocator())

    if show_grid:
        ax.grid(True, which='major', axis='y', linewidth=0.4)
        if show_minor_y:
            ax.grid(True, which='minor', axis='y', linewidth=0.15, alpha=0.3)
    else:
        ax.grid(False)

    major_tick_width = 1.2 if label_weight == 'bold' else 0.8
    major_tick_length = 6 if label_weight == 'bold' else 4
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=tick_label_size,
        width=major_tick_width,
        length=major_tick_length,
        color=axis_color,
        labelcolor=tick_color,
    )
    ax.tick_params(axis='both', which='minor', width=0.8, length=3, color=axis_color)

    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight(label_weight)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(axis_color)
        ax.spines[spine].set_linewidth(major_tick_width)


def add_legend(ax):
    legend_elements = [
        Line2D([0], [0], marker='o', color='white', markerfacecolor=DFT_COLOR,
               markeredgecolor=DFT_COLOR, markersize=7, label='True KS-DFT'),
        Line2D([0], [0], marker='o', color='white', markerfacecolor=MLIP_COLOR,
               markeredgecolor=MLIP_COLOR, markersize=7, label='DFT-trained MLIP'),
        Line2D([0], [0], marker='D', color='white', markerfacecolor=THIS_COLOR,
               markeredgecolor='#CC3355', markersize=7, label='This work'),
        Line2D([0], [0], marker='*', color='white', markerfacecolor=DFT_COLOR,
               markersize=10, label='Heroic run'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        fontsize=7,
        framealpha=0.9,
        facecolor='white',
        edgecolor='#CCCCCC',
        labelcolor='#333333',
    )


def save_figure(fig, stem):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    png_path = RESULTS_DIR / f'{stem}.png'
    svg_path = RESULTS_DIR / f'{stem}.svg'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white', format='svg')
    plt.close(fig)
    return png_path, svg_path


def build_detailed_plot():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(dft_env_y, dft_env_a, '-', color=DFT_COLOR, alpha=0.3, linewidth=1.5, zorder=1)
    ax.axhspan(100, 500, color=DFT_COLOR, alpha=0.04, zorder=0)
    ax.text(
        2028.8,
        250,
        'Routine KS-DFT\nceiling (~300 atoms)',
        fontsize=6,
        color=DFT_COLOR,
        alpha=0.7,
        va='center',
        style='italic',
        ha='center',
    )
    plot_dft_points(ax)

    ax.plot(mlip_env_y, mlip_env_a, '--', color=MLIP_COLOR, alpha=0.4, linewidth=1.8, zorder=1)
    plot_mlip_points(ax)

    add_manual_label(
        ax,
        (2026, 612000),
        'This work\n250k-612k atoms\n~40 ts/s, 256 GPUs',
        (2027.0, 4_000_000),
        color=THIS_COLOR,
        ha='left',
        va='bottom',
        fontsize=8,
        fontweight='bold',
        line_color=THIS_COLOR,
        arrowstyle='->',
        line_width=1.3,
        line_alpha=1.0,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=THIS_COLOR, lw=1.2),
        zorder=15,
    )

    # Edit the `text_xy` tuples below to manually move labels around.
    mlip_annots = [
        (2021, 113246208, '113M atoms, 1 ns/day\n(DeePMD, Summit)', (2020.4, 2.3e8), 'right', 'bottom'),
        (2025.15, 1024000000, '1B atoms\n(tabGAP, W cascades)', (2024.7, 2.2e9), 'right', 'bottom'),
        (2023, 100640512, '100.6M atoms\n(Allegro)', (2022.5, 6.0e7), 'right', 'top'),
        (2023.15, 8100000, 'W cascades, 8.1M\n(1 GPU)', (2022.4, 4.0e6), 'right', 'center'),
        (2025, 27648000, 'HEA cascades\n27.6M atoms', (2024.2, 5.2e7), 'right', 'center'),
    ]
    for yr, at, label, text_xy, ha_side, va_side in mlip_annots:
        add_manual_label(
            ax,
            (yr, at),
            label,
            text_xy,
            color='#1B6B2B',
            line_color=MLIP_COLOR,
            fontsize=6,
            ha=ha_side,
            va=va_side,
        )

    dft_annots = [
        (2009, 32768, '32k atoms\n(linear-scaling DFT)', (2010.0, 7.0e4), 'left', 'bottom'),
        (2014, 5898, '~6k atoms\n(VASP, 9k cores)', (2013.2, 1.3e4), 'right', 'bottom'),
        (2023.15, 3584, '3,584 atoms\n(heroic GPU run)', (2024.0, 1.1e4), 'left', 'bottom'),
    ]
    for yr, at, label, text_xy, ha_side, va_side in dft_annots:
        add_manual_label(
            ax,
            (yr, at),
            label,
            text_xy,
            color='#2B5588',
            line_color=DFT_COLOR,
            fontsize=5.5,
            ha=ha_side,
            va=va_side,
            line_width=0.5,
            line_alpha=0.35,
        )

    milestones = [
        (1999, 'Born'),
        (2010, 'First\ncomputer'),
        (2018, 'Started\nMIT'),
        (2022, 'Started\nPhD'),
        (2026, 'Finished\nPhD'),
    ]
    for yr, label in milestones:
        ax.axvline(yr, color='#BBBBBB', alpha=0.5, linewidth=0.7, linestyle=':', zorder=0)
        ax.text(
            yr,
            22,
            label,
            fontsize=6.5,
            color='#777777',
            ha='center',
            va='top',
            style='italic',
            fontweight='bold',
        )

    ax.annotate(
        '',
        xy=(2025.5, 800_000_000),
        xytext=(2000, 300),
        arrowprops=dict(
            arrowstyle='->',
            color=ACCENT_GRAY,
            lw=1.8,
            alpha=0.15,
            connectionstyle='arc3,rad=0.15',
        ),
    )
    ax.text(
        2009.5,
        50_000_000,
        '>10⁶× expansion\nin capability',
        fontsize=11,
        color='#555555',
        alpha=0.4,
        fontweight='bold',
        rotation=28,
        ha='center',
    )

    style_axes(
        ax,
        y_label=DEFAULT_Y_LABEL,
        xlim=(1997, 2031.5),
        x_label_size=10,
        y_label_size=9,
        tick_label_size=8,
    )
    add_legend(ax)
    fig.tight_layout(pad=0.8)
    return fig


def build_simple_plot():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    plot_dft_points(ax, scale=1.25)
    plot_mlip_points(ax, scale=1.25)
    style_axes(
        ax,
        y_label=SIMPLE_Y_LABEL,
        xlim=(1997, 2030.5),
        x_label_size=24,
        y_label_size=22,
        tick_label_size=16,
        label_weight='bold',
        axis_color='#555555',
        tick_color='#222222',
        show_grid=False,
        show_minor_x=False,
        show_minor_y=False,
    )
    fig.tight_layout(pad=1.0)
    return fig


def main():
    detailed_png, detailed_svg = save_figure(build_detailed_plot(), DETAILED_OUTPUT_STEM)
    simple_png, simple_svg = save_figure(build_simple_plot(), SIMPLE_OUTPUT_STEM)
    print(f'Saved detailed plot to {detailed_png} and {detailed_svg}.')
    print(f'Saved simple plot to {simple_png} and {simple_svg}.')


if __name__ == '__main__':
    main()
