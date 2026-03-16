"""
Shared publication-quality plotting utilities.

Provides a colorblind-safe palette (Tol Bright), consistent rcParams,
standard figure sizes, and a dual-format save function (PNG + PDF).

Usage (from any script inside scripts/):
    import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
    from plotting_utils import set_pub_style, save_fig, TOL_BRIGHT, DOUBLE_COL

Or more explicitly:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plotting_utils import set_pub_style, save_fig, TOL_BRIGHT, DOUBLE_COL, SINGLE_COL, fig_size
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------------------------------------------------------------------
# Tol Bright palette  (Paul Tol, https://personal.sron.nl/~pault/)
# 7 colours designed for deuteranopia / protanopia safety with high contrast.
# ---------------------------------------------------------------------------
TOL_BRIGHT = [
    "#4477AA",  # blue
    "#EE6677",  # red / coral
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
]

# Semantic-name access
COLORS = {
    "blue":   TOL_BRIGHT[0],
    "red":    TOL_BRIGHT[1],
    "green":  TOL_BRIGHT[2],
    "yellow": TOL_BRIGHT[3],
    "cyan":   TOL_BRIGHT[4],
    "purple": TOL_BRIGHT[5],
    "grey":   TOL_BRIGHT[6],
}


def get_color(index: int) -> str:
    """Return a palette colour by index (wraps around)."""
    return TOL_BRIGHT[index % len(TOL_BRIGHT)]


# ---------------------------------------------------------------------------
# Standard figure dimensions (inches)
# ---------------------------------------------------------------------------
DOUBLE_COL = 7.0   # generic double-column width
SINGLE_COL = 3.5   # single-column / half-width panel
GOLDEN = 0.618     # golden-ratio aspect multiplier


def fig_size(width: float = DOUBLE_COL, aspect: float = GOLDEN) -> tuple:
    """Return (width, height) tuple in inches."""
    return (width, width * aspect)


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
def set_pub_style(base_fontsize: int = 10, **overrides):
    """
    Apply publication-quality matplotlib rcParams.

    Parameters
    ----------
    base_fontsize : int
        Root font size; everything else scales from this.
    **overrides : dict
        Any extra rcParams to set (applied last, so they win).
    """
    # --- font fallback chain ---
    available = {f.name for f in fm.fontManager.ttflist}
    if "Helvetica" in available:
        preferred = "Helvetica"
    elif "Arial" in available:
        preferred = "Arial"
    else:
        preferred = "DejaVu Sans"

    from cycler import cycler

    params = {
        # Font
        "font.family":       "sans-serif",
        "font.sans-serif":   [preferred],
        "font.size":         base_fontsize,
        "axes.titlesize":    base_fontsize + 2,
        "axes.labelsize":    base_fontsize + 1,
        "xtick.labelsize":   base_fontsize,
        "ytick.labelsize":   base_fontsize,
        "legend.fontsize":   base_fontsize - 1,
        "figure.titlesize":  base_fontsize + 4,

        # PDF / PS font embedding (avoid Type-3 fonts)
        "pdf.fonttype":  42,
        "ps.fonttype":   42,

        # Axes
        "axes.linewidth":    1.2,
        "axes.prop_cycle":   cycler(color=TOL_BRIGHT),

        # Ticks
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.size":  5,
        "ytick.major.size":  5,
        "xtick.minor.size":  3,
        "ytick.minor.size":  3,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,

        # Lines / markers
        "lines.linewidth":   1.5,
        "lines.markersize":  5,

        # Legend
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "0.8",
        "legend.fancybox":      False,

        # Figure
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "figure.figsize":    list(fig_size()),
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",

        # Grid (off by default; scripts can enable per-plot)
        "axes.grid":         False,
    }

    params.update(overrides)
    matplotlib.rcParams.update(params)


# ---------------------------------------------------------------------------
# Dual-format figure saving
# ---------------------------------------------------------------------------
def save_fig(fig, path_stem, dpi: int = 300, tight: bool = True):
    """
    Save a figure as both high-resolution PNG and vector PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path_stem : str or Path
        File path *without* extension. Two files are written:
        ``{path_stem}.png`` and ``{path_stem}.pdf``.
    dpi : int
        Resolution for the PNG raster output.
    tight : bool
        Whether to use bbox_inches='tight'.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(str(stem.with_suffix(".png")), dpi=dpi, bbox_inches=bbox)
    fig.savefig(str(stem.with_suffix(".pdf")), bbox_inches=bbox)
    print(f"  Saved: {stem.with_suffix('.png')}  &  {stem.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Optional seaborn integration
# ---------------------------------------------------------------------------
def set_sns_style():
    """
    Configure seaborn to use the Tol Bright palette and a compatible theme.

    Call *after* ``set_pub_style()`` so that font rcParams are inherited.
    """
    try:
        import seaborn as sns
        sns.set_palette(TOL_BRIGHT)
        sns.set_style("ticks", {
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
        })
    except ImportError:
        pass  # seaborn not installed; silently skip
