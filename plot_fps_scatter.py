"""Generate a paper-quality FPS scatter figure for the MipNeRF-360 comparison.

Produces a single combined figure with three aligned panels:
  - Top-right:    2D scatter of methods with both raster and ray FPS.
  - Top-left:     1D (vertical) scatter of rasterization-only methods, sharing
                  the y-axis (Rasterization FPS) with the 2D panel.
  - Bottom-right: 1D (horizontal) scatter of ray-tracing-only methods, sharing
                  the x-axis (Ray Tracing FPS) with the 2D panel.

Design intent: the 1D panels look 1D (a single axis line with labelled dots,
no background, ticks, grid, or extra spines). The 2D panel keeps only the
elements needed to read it: axes, the "better" region, dots, halo on ours,
and category labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec


# Register the Jost faces the webpage uses so the plot typography matches
# the surrounding HTML / video overlays. The variable font supplies normal
# weight; Jost-Black is registered separately so matplotlib has a heavy
# face it can pick up when fontweight="bold" is requested (matplotlib
# resolves fontweight by closest-weight lookup across registered faces).
_FONT_DIR = Path(__file__).resolve().parent / "static" / "fonts"
for _name in ("Jost-VariableFont.ttf", "Jost-Black.ttf"):
    _p = _FONT_DIR / _name
    if _p.exists():
        font_manager.fontManager.addfont(str(_p))


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
    return dict(color=OURS_COLOR, marker="*", s=360, zorder=5,
                edgecolors="black", linewidths=1.0)


def _other_style() -> dict:
    return dict(color=OTHER_COLOR, marker="o", s=130, zorder=4,
                edgecolors="white", linewidths=1.0)


def _point_style(name: str) -> dict:
    return _ours_style() if _is_ours(name) else _other_style()


def _halo_around(ax, xy, size=650) -> None:
    ax.scatter(*xy, s=size, color=OURS_HALO, alpha=0.55,
               edgecolors="none", zorder=4, marker="*")


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
        fontsize=12,
        fontweight="bold" if _is_ours(name) else "normal",
        color=OURS_COLOR if _is_ours(name) else "black",
    )


def _strip_panel(ax) -> None:
    """Hide every spine, tick mark, grid line, and background fill on `ax`.

    We do *not* clear the underlying tick locations so that any shared axes
    (e.g. the 2D main panel) still see them.
    """
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(
        which="both",
        left=False, right=False, top=False, bottom=False,
        labelleft=False, labelright=False,
        labeltop=False, labelbottom=False,
        length=0,
    )
    ax.grid(False)


# ---------------------------------------------------------------------------
# 1D panels
# ---------------------------------------------------------------------------


def _compute_placements(values, close_threshold):
    """For each sorted value, decide whether its label sits on the "positive"
    side (above / right) or "negative" side (below / left). When two values
    are close together we spread them outward: the lower one moves to the
    negative side, the upper one stays on the positive side.
    """
    placements = ["pos"] * len(values)
    for i in range(len(values) - 1):
        if (values[i + 1] - values[i]) < close_threshold:
            placements[i] = "neg"
            placements[i + 1] = "pos"
    return placements


def _draw_horizontal_1d(ax, entries, close_threshold, x_lim,
                        default_above=True, line_y=0.0) -> None:
    # The single horizontal line *is* the 1D axis. `line_y` slides it up/down
    # within the panel to control how close the 1D dots sit to the 2D plot.
    ax.plot(x_lim, [line_y, line_y], color=AXIS_LINE_COLOR, linewidth=1.2,
            zorder=1)
    sorted_entries = sorted(entries, key=lambda e: e[1])
    placements = _compute_placements([e[1] for e in sorted_entries], close_threshold)
    for (name, fps), placement in zip(sorted_entries, placements):
        ax.scatter(fps, line_y, **_point_style(name))
        # When default_above is False we flip the convention so labels sit
        # below the axis line by default.
        above = (placement == "pos") if default_above else (placement == "neg")
        _annotate(
            ax,
            xy=(fps, line_y),
            text=_label_text(name, f"{fps} FPS"),
            name=name,
            offset=(0, 14 if above else -14),
            ha="center",
            va="bottom" if above else "top",
        )


def _draw_vertical_1d(ax, entries, close_threshold, y_lim, line_x=0.0) -> None:
    ax.plot([line_x, line_x], y_lim, color=AXIS_LINE_COLOR, linewidth=1.2,
            zorder=1)
    sorted_entries = sorted(entries, key=lambda e: e[1])
    placements = _compute_placements([e[1] for e in sorted_entries], close_threshold)
    for (name, fps), placement in zip(sorted_entries, placements):
        ax.scatter(line_x, fps, **_point_style(name))
        above = placement == "pos"
        _annotate(
            ax,
            xy=(line_x, fps),
            text=_label_text(name, f"{fps} FPS"),
            name=name,
            offset=(-14, 10 if above else -10),
            ha="right",
            va="bottom" if above else "top",
        )


# ---------------------------------------------------------------------------
# 2D panel
# ---------------------------------------------------------------------------


def _draw_2d(ax, entries, x_max, y_max) -> None:
    # Per-point label placement to avoid overlapping neighbours.
    # Each entry: (ha, va, offset_px).
    label_placement = {
        "Power Foam (ours)": ("right", "bottom", (-16, 12)),
        "3DGUT": ("left", "bottom", (12, 12)),
        "Radiance Meshes*": ("left", "top", (12, -12)),
    }
    for name, ray_fps, raster_fps in entries:
        if _is_ours(name):
            _halo_around(ax, (ray_fps, raster_fps), size=900)
        ax.scatter(ray_fps, raster_fps, **_point_style(name))
        ha, va, offset = label_placement.get(name, ("left", "bottom", (12, 12)))
        _annotate(
            ax,
            xy=(ray_fps, raster_fps),
            text=_label_text(name, f"ray {ray_fps} / raster {raster_fps}"),
            name=name,
            offset=offset,
            ha=ha,
            va=va,
        )



def _category_label(ax, text, x, y, ha="center", va="center", rotation=0) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=14,
        color="#334155",
        fontweight="bold",
        # Note: 14pt bold renders heavier than the 12pt body labels below
        # while staying smaller than the 18pt axis labels — same visual
        # hierarchy as <h3> in the page (subsection title).
        rotation=rotation,
    )


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def build_figure(out_path: Path) -> None:
    # Sizes are tuned so the rasterised plot, when displayed inside the
    # page's `.container.is-max-desktop` (~1000 px wide on desktop) maps
    # to roughly the same on-screen pixel sizes as the surrounding HTML:
    # body labels ≈ 18 px, headings/axis labels ≈ 28 px. The image is
    # saved at dpi=600 (~5 px/pt at the typical 0.2x display scale), so
    # body text needs ≈ 12 pt and headings ≈ 18 pt in matplotlib units.
    plt.rcParams.update({
        "font.family": ["Jost", "sans-serif"],
        "font.size": 12,
        "axes.labelcolor": "#1e293b",
        "axes.labelsize": 18,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "text.antialiased": True,
        # Render mathtext (e.g. r"$\beta$-splat") with sans-serif glyphs so
        # the Greek letter blends in instead of looking like Computer Modern.
        "mathtext.fontset": "dejavusans",
    })

    all_ray = [r for _, r, _ in BOTH] + [r for _, r in RAYTRACING_ONLY]
    all_raster = [s for _, _, s in BOTH] + [r for _, r in RASTERIZATION_ONLY]
    x_max = max(all_ray) * 1.2
    y_max = max(all_raster) * 1.12
    x_min, y_min = 0.0, 0.0

    fig = plt.figure(figsize=(11, 8.8))
    fig.patch.set_facecolor("white")
    gs = GridSpec(
        nrows=2,
        ncols=2,
        width_ratios=[1.6, 4],
        height_ratios=[4, 1.4],
        wspace=0.0,
        hspace=0.0,
    )

    ax_main = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)

    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    # --- Main 2D panel ---
    # Keep only the L-shaped axes (left + bottom). Drop fill, grid, and the
    # top/right spines so nothing competes with the data. Numerical tick
    # labels stay on the main panel (the 1D panels are stripped of ticks).
    ax_main.set_facecolor("none")
    for side in ("top", "right"):
        ax_main.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_main.spines[side].set_color(AXIS_LINE_COLOR)
        ax_main.spines[side].set_linewidth(1.2)
    ax_main.tick_params(
        which="both", labelbottom=True, labelleft=True,
        length=4, color=AXIS_LINE_COLOR,
    )
    from matplotlib.ticker import MultipleLocator
    ax_main.xaxis.set_major_locator(MultipleLocator(50))
    ax_main.yaxis.set_major_locator(MultipleLocator(50))
    _draw_2d(ax_main, BOTH, x_max, y_max)
    _category_label(
        ax_main,
        "Rasterization + Ray Tracing",
        x=0.5, y=1.02, va="bottom",
    )

    # --- Left vertical 1D panel ---
    _strip_panel(ax_left)
    ax_left.set_xlim(-1, 1)
    raster_range = max(r for _, r in RASTERIZATION_ONLY) - min(r for _, r in RASTERIZATION_ONLY)
    # Match the visible gap between the rasterization 1D line and the 2D
    # plot's y-axis to the gap between the ray-tracing 1D line and the 2D
    # plot's x-axis. With the current panel aspect ratios this works out to
    # line_x ≈ 0.42 in the panel's [-1, 1] x range.
    _draw_vertical_1d(
        ax_left, RASTERIZATION_ONLY,
        close_threshold=max(raster_range * 0.12, 12),
        y_lim=(y_min, y_max),
        line_x=0.42,
    )
    ax_left.set_ylabel("Rasterization FPS", fontsize=18)
    # Pull the y-axis title in closer (default sits near the figure edge),
    # but leave a small breathing margin past the data labels.
    ax_left.yaxis.set_label_coords(0.18, 0.5)

    # --- Bottom horizontal 1D panel ---
    _strip_panel(ax_bottom)
    ax_bottom.set_ylim(-1, 1)
    ray_range = max(r for _, r in RAYTRACING_ONLY) - min(r for _, r in RAYTRACING_ONLY)
    # Slide the ray-tracing 1D line 20% closer to the 2D plot. The panel's
    # y range is [-1, 1]; line at y=0.2 reduces the original gap by 20%.
    _draw_horizontal_1d(
        ax_bottom, RAYTRACING_ONLY,
        close_threshold=max(ray_range * 0.12, 8),
        x_lim=(x_min, x_max),
        default_above=False,
        line_y=0.2,
    )
    ax_bottom.set_xlabel("Ray Tracing FPS", fontsize=18)
    # Pull the x-axis title up so it sits just below the data labels rather
    # than at the bottom of the cropped figure. Axes coords: 0 = panel
    # bottom, 1 = panel top; data labels live near axes y≈0.5.
    ax_bottom.xaxis.set_label_coords(0.5, 0.15)

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
