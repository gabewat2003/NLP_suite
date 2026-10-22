"""One matplotlib style for every committed figure.

Colors are the first slots of a CVD-validated categorical palette, assigned in
fixed order (series 1 is always blue, series 2 orange, series 3 aqua).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9890"


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2, "lines.markersize": 7, "legend.frameon": False,
        "legend.fontsize": 9, "font.size": 10, "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def save(fig, name):
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
