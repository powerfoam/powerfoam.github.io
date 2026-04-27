"""Render `static/videos/powerfoam-title.mp4` -- a perfectly-looping
animated Power Foam wordmark.

Idea: same blue-noise + power-diagram pipeline as `title.py`, but every
frame perturbs each cell's position and radius by a small periodic
offset (sin/cos of `2π * t / T` so the loop is seamless). Each cell
gets its own random phase so the motion looks organic rather than a
synchronised wobble. We recompute the power diagram every frame so the
white separators between cells follow the cells correctly.

Frames are rasterised with matplotlib (Agg) and encoded to H.264 with
the ffmpeg shipped by `imageio_ffmpeg`. Output is a small (~few MB),
even-dimension yuv420p MP4 that browsers will autoplay-loop just like
the dipole videos already on the page.
"""

from __future__ import annotations

import io
import sys
import random
import colorsys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Arc, Circle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from visualize_power_diagram import (  # noqa: E402
    compute_power_diagram_edges,
    generate_blue_noise_points,
)


# ---------------------------------------------------------------------------
# Output / encoding ----------------------------------------------------------

OUT_DIR = ROOT / "static/videos"
OUT_MP4 = OUT_DIR / "powerfoam-title.mp4"      # white-bg fallback
OUT_WEBM = OUT_DIR / "powerfoam-title.webm"    # alpha-preserving primary
VIDEO_WIDTH_PX = 1024          # final width before x264 even-dim padding
DPI = 100
FPS = 30
DURATION_S = 4.0               # one full motion cycle
N_FRAMES = int(round(FPS * DURATION_S))

# Encoding -- aim for "small, fast-decoding, looks fine" rather than maximum
# fidelity. yuv420p + faststart means it autoplays everywhere.
CRF = 22
PRESET = "slow"

# ---------------------------------------------------------------------------
# Cell sampling (mirrors title.py) -----------------------------------------

TEXT = "Power Foam"
SEED = 42
TEXT_FONT_SIZE = 120
FIGSIZE = (12, 3)
RAW_DPI = 100
BLUE_NOISE_R = 7.5
RADIUS_MIN, RADIUS_MAX = 6.0, 10.0
NOISE_STD = 0.1

# ---------------------------------------------------------------------------
# Motion --------------------------------------------------------------------

# Each cell oscillates around its base position by ±POS_AMPLITUDE units in
# a Lissajous pattern (sin in x, cos in y) and around its base radius by
# ±RAD_AMPLITUDE units. The pattern uses 2π * t/T so frame 0 == frame N
# and the loop is seamless.
POS_AMPLITUDE = 1.6
RAD_AMPLITUDE = 1.0

EDGE_WIDTH = 1.2
EDGE_COLOR = "white"
# Frames are rendered with a transparent background -- the WebM output
# preserves the alpha channel (libvpx-vp9 + yuva420p) and the MP4 fallback
# composites against `MP4_BG_COLOR` so it still works in browsers that
# can't play VP9 alpha.
MP4_BG_COLOR = "white"


def _hsv_color(rgb_8bit, sat=0.5, val=0.9):
    r, g, b = (c / 255.0 for c in rgb_8bit)
    h, _, _ = colorsys.rgb_to_hsv(r, g, b)
    return colorsys.hsv_to_rgb(h, sat, val)


COLOR_START = _hsv_color((16, 0, 126))
COLOR_END = _hsv_color((145, 21, 21))


def render_text_mask() -> np.ndarray:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fp = FontProperties(family="sans-serif", weight="bold")
    ax.text(0.5, 0.5, TEXT, fontsize=TEXT_FONT_SIZE,
            ha="center", va="center", fontproperties=fp)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=RAW_DPI, bbox_inches="tight",
                pad_inches=0.1, transparent=False)
    buf.seek(0)
    plt.close(fig)
    img = mpimg.imread(buf)
    if img.ndim == 3:
        return np.mean(img[..., :3], axis=2)
    return img


def sample_cells(gray: np.ndarray):
    H, W = gray.shape
    y_idx, x_idx = np.where(gray < 0.5)
    min_x, max_x = int(x_idx.min()), int(x_idx.max())
    min_y, max_y = int(y_idx.min()), int(y_idx.max())

    raw = generate_blue_noise_points(min_x, max_x, min_y, max_y, r=BLUE_NOISE_R)
    xs, ys = [], []
    for p in raw:
        px, py = int(p[0]), int(p[1])
        if 0 <= py < H and 0 <= px < W and gray[py, px] < 0.5:
            xs.append(p[0])
            ys.append(-p[1])  # math y-up
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    if NOISE_STD > 0:
        xs += np.random.normal(0, NOISE_STD, size=xs.shape)
        ys += np.random.normal(0, NOISE_STD, size=ys.shape)
    xs -= xs.mean()
    ys -= ys.mean()

    cmap = LinearSegmentedColormap.from_list("pf", [COLOR_START, COLOR_END])
    norm_x = (xs - xs.min()) / (xs.max() - xs.min() + 1e-6)
    cols = cmap(norm_x)[:, :3]
    radii = np.random.uniform(RADIUS_MIN, RADIUS_MAX, size=len(xs))
    return np.column_stack([xs, ys]), radii, cols


# ---------------------------------------------------------------------------
# Frame rendering ----------------------------------------------------------

def render_frame(ax, points, radii, colors):
    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    # Transparent frame background: the WebM encoder will preserve the
    # alpha; the MP4 encoder will composite against white via ffmpeg.
    ax.set_facecolor((0, 0, 0, 0))

    for pt, r, c in zip(points, radii, colors):
        ax.add_patch(Circle(pt, r, color=c, ec="none"))

    edges, _, _, arcs, _ = compute_power_diagram_edges(points, radii)

    restricted = [[e["p1"], e["p2"]] for e in edges if e["type"] == "restricted"]
    if restricted:
        ax.add_collection(LineCollection(
            restricted, colors=EDGE_COLOR, linewidths=EDGE_WIDTH, zorder=2
        ))
    for arc in arcs:
        ax.add_patch(Arc(
            arc["center"], 2 * arc["radius"], 2 * arc["radius"],
            angle=0,
            theta1=arc["start_angle"],
            theta2=arc["end_angle"],
            color=EDGE_COLOR,
            linewidth=EDGE_WIDTH,
            zorder=2,
        ))


def main() -> None:
    np.random.seed(SEED)
    random.seed(SEED)

    gray = render_text_mask()
    base_pts, base_radii, colors = sample_cells(gray)
    N = len(base_pts)
    print(f"sampled {N} cells")

    # Per-cell phases so the motion isn't synchronised. Drawn after the
    # blue-noise step so the same SEED still gives the same base layout.
    phase_x = np.random.uniform(0, 2 * np.pi, N)
    phase_y = np.random.uniform(0, 2 * np.pi, N)
    phase_r = np.random.uniform(0, 2 * np.pi, N)

    # Static viewport: pad by the largest possible cell extent so a cell
    # at peak amplitude + peak radius never clips at the edges. This also
    # ensures every frame uses the same axis limits, which keeps the
    # encoded video's content stable in pixel space.
    max_excursion = POS_AMPLITUDE + (base_radii.max() + RAD_AMPLITUDE)
    pad = max_excursion * 1.05
    x0 = float(base_pts[:, 0].min()) - pad
    x1 = float(base_pts[:, 0].max()) + pad
    y0 = float(base_pts[:, 1].min()) - pad
    y1 = float(base_pts[:, 1].max()) + pad
    aspect = (x1 - x0) / (y1 - y0)
    fig_w_in = VIDEO_WIDTH_PX / DPI
    fig_h_in = fig_w_in / aspect

    print(f"viewport: ({x0:.1f}, {y0:.1f}) .. ({x1:.1f}, {y1:.1f})")
    print(f"figure: {fig_w_in:.2f} x {fig_h_in:.2f} in @ {DPI} dpi  "
          f"(~{int(fig_w_in*DPI)}x{int(fig_h_in*DPI)} px)")
    print(f"frames: {N_FRAMES} @ {FPS} fps  ({DURATION_S}s loop)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # One figure reused across all frames; ax.clear() is much cheaper
        # than re-creating subplots (font/GC churn).
        fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=DPI)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.patch.set_alpha(0.0)  # transparent figure background

        for f in range(N_FRAMES):
            t = 2 * np.pi * f / N_FRAMES  # one full cycle across N_FRAMES
            dx = POS_AMPLITUDE * np.sin(t + phase_x)
            dy = POS_AMPLITUDE * np.cos(t + phase_y)
            dr = RAD_AMPLITUDE * np.sin(t + phase_r)

            pts = base_pts + np.column_stack([dx, dy])
            radii = base_radii + dr

            render_frame(ax, pts, radii, colors)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

            out = tmp / f"f_{f:04d}.png"
            fig.savefig(out, dpi=DPI, transparent=True, edgecolor="none")
            if f % 15 == 0:
                print(f"  rendered frame {f+1}/{N_FRAMES}")
        plt.close(fig)

        # ---- Encode with the bundled ffmpeg --------------------------
        import imageio_ffmpeg
        ffp = imageio_ffmpeg.get_ffmpeg_exe()

        # 1) WebM (VP9 + alpha). Modern Chrome / Firefox / Safari 16+
        #    play this with a transparent background, so the title sits
        #    flush against whatever colour the page hero happens to be.
        webm_cmd = [
            ffp, "-y",
            "-framerate", str(FPS),
            "-i", str(tmp / "f_%04d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=#00000000",
            "-an",
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-b:v", "0",
            "-crf", "32",
            "-row-mt", "1",
            "-deadline", "good",
            "-cpu-used", "2",
            str(OUT_WEBM),
        ]
        print("encoding WebM (alpha):", OUT_WEBM.name)
        subprocess.run(webm_cmd, check=True)

        # 2) MP4 (H.264) fallback for older browsers -- no alpha, so we
        #    composite the frames against MP4_BG_COLOR explicitly.
        mp4_cmd = [
            ffp, "-y",
            "-framerate", str(FPS),
            "-i", str(tmp / "f_%04d.png"),
            "-vf",
            f"pad=ceil(iw/2)*2:ceil(ih/2)*2:color={MP4_BG_COLOR}",
            "-an",
            "-c:v", "libx264",
            "-preset", PRESET,
            "-crf", str(CRF),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(OUT_MP4),
        ]
        print("encoding MP4 (white bg):", OUT_MP4.name)
        subprocess.run(mp4_cmd, check=True)

    for p in (OUT_WEBM, OUT_MP4):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
