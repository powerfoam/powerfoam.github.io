"""Generate a paper-quality FPS scatter figure for the MipNeRF-360 comparison.

Single 2D axes:
  - Methods with both raster and ray FPS sit at their (ray, raster) coordinate.
  - Rasterization-only methods (no ray tracing) sit on the y-axis (x = 0).
  - Ray-tracing-only methods (no rasterization) sit on the x-axis (y = 0).
  - The "0" tick on each axis is relabelled "NA" since x=0 / y=0 here means
    "this method has no ray-tracing / rasterization number, not literally 0 FPS".

Design intent: keep only the elements needed to read the plot: the L-shaped
axes, the dots, a halo on ours, the category labels, and "NA" at the origin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MultipleLocator


# Match the page typography. index.css declares `font-family: 'Jost',
# sans-serif` and the variable TTF is bundled under static/fonts/. We
# register it with matplotlib so the figure inherits the same look as
# the rest of the webpage.
REPO_ROOT = Path(__file__).resolve().parent.parent
JOST_FONT_PATH = REPO_ROOT / "static" / "fonts" / "Jost-VariableFont.ttf"


def _register_page_font() -> str:
    """Register the bundled Jost variable font with matplotlib and return
    the family name to use in `font.family`. Falls back to 'sans-serif'
    if the file is missing for some reason.
    """
    if not JOST_FONT_PATH.is_file():
        return "sans-serif"
    font_manager.fontManager.addfont(str(JOST_FONT_PATH))
    return font_manager.FontProperties(fname=str(JOST_FONT_PATH)).get_name()


RASTERIZATION_ONLY = [
    ("3DGS", 293),
    ("3DGS-MCMC", 302),
    (r"$\beta$-splat", 137),
]

RAYTRACING_ONLY = [
    ("3DGRT", 64),
    ("Radiant Foam", 180),
]

BOTH = [
    ("3DGUT", 55, 177),
    ("Radiance Meshes*", 118, 159),
    ("Power Foam (ours)", 174, 275),
]

# Paper-friendly palette.
OURS_COLOR = "#e63946"
OURS_HALO = "#f4a4ac"
OTHER_COLOR = "#1d3557"
AXIS_LINE_COLOR = "#475569"


def _is_ours(name: str) -> bool:
    return "ours" in name.lower()


def _ours_style() -> dict:
    return dict(color=OURS_COLOR, marker="*", s=620, zorder=5,
                edgecolors="black", linewidths=1.2)


def _other_style() -> dict:
    return dict(color=OTHER_COLOR, marker="o", s=130, zorder=4,
                edgecolors="white", linewidths=1.0)


def _point_style(name: str) -> dict:
    return _ours_style() if _is_ours(name) else _other_style()


def _halo_around(ax, xy, size=650) -> None:
    ax.scatter(*xy, s=size, color=OURS_HALO, alpha=0.55,
               edgecolors="none", zorder=4, marker="*")


def _ours_guide_lines(ax, xy) -> None:
    """Dashed red guide lines from the "ours" star down to the x-axis
    and across to the y-axis. Drawn at low zorder so they sit behind
    the star and its halo.
    """
    x, y = xy
    dash_kwargs = dict(
        linestyle=(0, (6, 5)),
        linewidth=1.8,
        color=OURS_COLOR,
        alpha=0.75,
        solid_capstyle="butt",
        dash_capstyle="butt",
        # Sit below every annotation (text default zorder is 3) so the
        # dashes never run over the per-point labels.
        zorder=0,
    )
    ax.plot([x, x], [0, y], **dash_kwargs)
    ax.plot([0, x], [y, y], **dash_kwargs)


def _label_text(name: str, value_str: str) -> str:
    return f"{name}\n{value_str}"


def _annotate(ax, xy, text, name, offset, ha="center", va="bottom") -> None:
    ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=14,
        fontweight="bold" if _is_ours(name) else "normal",
        color=OURS_COLOR if _is_ours(name) else "black",
        zorder=6,
    )


# ---------------------------------------------------------------------------
# 2D panel (now also hosts the 1D entries on its axes)
# ---------------------------------------------------------------------------


# Per-point label placement to avoid overlapping neighbours.
# Each entry: (ha, va, offset_px).
LABEL_PLACEMENT = {
    "Power Foam (ours)": ("right", "bottom", (-16, 12)),
    "3DGUT":             ("left",  "bottom", (12,  12)),
    "Radiance Meshes*":  ("left",  "top",    (12, -12)),
    # Rasterization-only (sit on the y-axis): label to the right of the dot.
    "3DGS-MCMC":         ("left",  "center", (14,  8)),
    "3DGS":              ("left",  "center", (14, -22)),
    r"$\beta$-splat":    ("left",  "center", (14,  0)),
    # Ray-tracing-only (sit on the x-axis): label above the dot.
    "3DGRT":             ("center", "bottom", (0, 14)),
    "Radiant Foam":      ("left",   "bottom", (16, 14)),
}


def _draw_point(ax, name, xy, full_label) -> None:
    if _is_ours(name):
        _ours_guide_lines(ax, xy)
        _halo_around(ax, xy, size=1500)
    ax.scatter(*xy, **_point_style(name))
    ha, va, offset = LABEL_PLACEMENT.get(name, ("left", "bottom", (12, 12)))
    _annotate(
        ax,
        xy=xy,
        text=_label_text(name, full_label),
        name=name,
        offset=offset,
        ha=ha,
        va=va,
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def build_figure(out_path: Path) -> None:
    page_font = _register_page_font()
    plt.rcParams.update({
        "font.family": page_font,
        "font.sans-serif": [page_font, "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 15,
        "axes.labelcolor": "#1e293b",
        "axes.labelsize": 16,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "text.antialiased": True,
        # Jost has no math glyphs, so render any $...$ math (e.g. the
        # β in "β-splat") with a sans-serif math fontset whose Greek
        # letters visually match the page font reasonably well.
        "mathtext.fontset": "dejavusans",
    })

    all_ray = [r for _, r, _ in BOTH] + [r for _, r in RAYTRACING_ONLY]
    all_raster = [s for _, _, s in BOTH] + [r for _, r in RASTERIZATION_ONLY]
    x_max = max(all_ray) * 1.25
    y_max = max(all_raster) * 1.15

    fig, ax = plt.subplots(figsize=(10, 7.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("none")

    # Leave a small negative margin so the dots that sit *on* the axes
    # (raster-only on x=0, ray-only on y=0) aren't clipped by the spines.
    x_pad = x_max * 0.02
    y_pad = y_max * 0.02
    ax.set_xlim(-x_pad, x_max)
    ax.set_ylim(-y_pad, y_max)

    # L-shaped axes only.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_LINE_COLOR)
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_position(("data", 0))

    # Arrowheads at the positive end of each axis. Drawn as triangle
    # markers in axes-fraction coords for x and y respectively, so they
    # follow the axis tip regardless of data limits / figure size.
    ax.plot(1, 0, marker=">", markersize=10,
            color=AXIS_LINE_COLOR, markeredgecolor=AXIS_LINE_COLOR,
            transform=ax.get_yaxis_transform(), clip_on=False, zorder=3)
    ax.plot(0, 1, marker="^", markersize=10,
            color=AXIS_LINE_COLOR, markeredgecolor=AXIS_LINE_COLOR,
            transform=ax.get_xaxis_transform(), clip_on=False, zorder=3)
    ax.tick_params(which="both", length=4, color=AXIS_LINE_COLOR,
                   labelbottom=True, labelleft=True)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(50))

    # The default 0 tick on each axis would draw two "0" labels (one
    # below the x-axis at x=0, one to the left of the y-axis at y=0).
    # Suppress both and write a single "0" annotation at the corner.
    def _hide_zero(value, _pos):
        if abs(value) < 1e-6:
            return ""
        return f"{int(round(value))}"
    ax.xaxis.set_major_formatter(FuncFormatter(_hide_zero))
    ax.yaxis.set_major_formatter(FuncFormatter(_hide_zero))
    # Single "N/A" sentinel in the third quadrant (bottom-left of the
    # origin). x=0 / y=0 in this figure means "this method cannot
    # rasterize / ray-trace", not literally 0 FPS.
    ax.annotate(
        "N/A",
        xy=(0, 0),
        xytext=(-8, -8), textcoords="offset points",
        ha="right", va="top",
        fontsize=13, color="#334155",
    )

    # Methods with both numbers: at (ray, raster).
    for name, ray_fps, raster_fps in BOTH:
        _draw_point(
            ax, name, (ray_fps, raster_fps),
            f"ray {ray_fps} / raster {raster_fps}",
        )

    # Rasterization-only methods: pinned to the y-axis (x = 0).
    for name, raster_fps in RASTERIZATION_ONLY:
        _draw_point(ax, name, (0, raster_fps), f"{raster_fps} FPS")

    # Ray-tracing-only methods: pinned to the x-axis (y = 0).
    for name, ray_fps in RAYTRACING_ONLY:
        _draw_point(ax, name, (ray_fps, 0), f"{ray_fps} FPS")

    ax.set_xlabel("Ray Tracing FPS", fontsize=19, labelpad=10)
    ax.set_ylabel("Rasterization FPS", fontsize=19, labelpad=14)

    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fps_plots",
        help="Directory where the plot will be written.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    build_figure(args.out_dir / "fps_combined.png")


if __name__ == "__main__":
    main()
