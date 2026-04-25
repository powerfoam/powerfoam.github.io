"""Label the four sites/bars in top.mp4 and displacement.mp4 with LaTeX
math-style labels (s_1..s_4 and d_1..d_4), colour-matched across the two
videos, then re-encode.

Pipeline:
  1) Extract every frame of displacement.mp4, find the 4 coloured bars,
     remember their x-centres and average colours.
  2) Render 4+4 transparent label PNGs via matplotlib mathtext (Computer
     Modern), each in the colour of its corresponding bar, with a thin
     white outline so the label reads on any background.
  3) Extract every frame of top.mp4, run dark-blob detection per frame
     (numpy + scipy.ndimage), greedy-permutation match against the
     previous frame so each detected dot keeps a stable identity, then
     paste the matching s_i label next to each dot.
  4) Overlay the d_i labels above each bar (static x positions) on
     every frame of displacement.mp4.
  5) Re-encode both labelled frame sequences with libx264.

Outputs:
  video/top_labeled.mp4
  video/displacement_labeled.mp4
"""

from __future__ import annotations

import io
import os
import subprocess
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
from matplotlib import patheffects, pyplot as plt, rcParams

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

ROOT = Path("video")
WORK = ROOT / "_overlay" / "work"
WORK.mkdir(parents=True, exist_ok=True)


def extract_frames(video: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(out_dir.iterdir()):
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(video),
             "-q:v", "2", str(out_dir / "frame_%04d.png")],
            check=True,
        )
    return sorted(out_dir.glob("frame_*.png"))


def detect_bars(frame: Path):
    """Return ([(cx, cy, mean_rgb, ymin, ymax)], axis_y) for each bar."""
    arr = np.asarray(Image.open(frame).convert("RGB"))
    H, W = arr.shape[:2]
    not_white = (arr.min(axis=2) < 235)
    not_black = (arr.max(axis=2) > 60)
    mask = not_white & not_black
    yi, xi = np.indices(mask.shape)
    mask &= xi > int(W * 0.22)
    mask &= (yi > int(H * 0.10)) & (yi < int(H * 0.90))
    labeled, n = ndimage.label(mask)
    bars = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) < 200:
            continue
        if (xs.max() - xs.min()) < 8 or (ys.max() - ys.min()) < 8:
            continue
        cx = xs.mean()
        cy = ys.mean()
        rgb = arr[ys, xs].mean(axis=0)
        bars.append((cx, cy, rgb, int(ys.min()), int(ys.max())))
    bars.sort(key=lambda b: b[0])

    # Detect the horizontal axis line: the row in the bar region with the
    # largest run of dark pixels.
    gray = np.asarray(Image.open(frame).convert("L"))
    axis_region = gray[:, int(W * 0.22):]
    dark_per_row = (axis_region < 80).sum(axis=1)
    axis_y = int(np.argmax(dark_per_row))
    return bars, axis_y


def render_label(text: str, color_rgb01: tuple[float, float, float],
                 fontsize: int = 26, dpi: int = 240,
                 stroke: float = 4.0) -> Image.Image:
    rcParams["mathtext.fontset"] = "cm"
    rcParams["mathtext.default"] = "it"
    fig = plt.figure(figsize=(1.6, 0.9))
    fig.patch.set_alpha(0)
    fig.text(
        0.5, 0.5, text, ha="center", va="center",
        fontsize=fontsize, color=color_rgb01,
        path_effects=[patheffects.withStroke(linewidth=stroke, foreground="white")],
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True,
                bbox_inches="tight", pad_inches=0.04, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    # Trim to the visible glyph; matplotlib leaves substantial vertical
    # padding (full font ascender/descender) which throws off any
    # geometric placement that uses the PNG's own dimensions.
    bbox = im.getbbox()
    if bbox is not None:
        im = im.crop(bbox)
    return im


def detect_dots(frame: Path) -> list[tuple[float, float]]:
    arr = np.asarray(Image.open(frame).convert("L"))
    H, W = arr.shape
    mask = arr < 70
    # Trim the outer 4 % so the disk silhouette and any rim antialiasing
    # never get picked up as a "blob".
    yi, xi = np.indices(mask.shape)
    mask &= (yi > int(H * 0.04)) & (yi < int(H * 0.96))
    mask &= (xi > int(W * 0.04)) & (xi < int(W * 0.96))
    labeled, n = ndimage.label(mask)
    dots: list[tuple[float, float]] = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if not (3 <= len(xs) <= 200):
            continue
        dots.append((float(xs.mean()), float(ys.mean())))
    return dots


def track(detections: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Greedy permutation matching to the previous frame.

    Falls back to "carry the previous position" for any frame where we
    don't see exactly 4 dots.
    """
    out: list[list[tuple[float, float]]] = []
    prev: list[tuple[float, float]] | None = None
    for dots in detections:
        if len(dots) != 4:
            if prev is None:
                # Pad with zeros so we always emit 4 IDs; first usable
                # frame will reseat them.
                dots = list(dots) + [(0.0, 0.0)] * (4 - len(dots))
            else:
                # Match what we do see, then fill the remaining IDs from prev.
                if dots:
                    used: set[int] = set()
                    matched = [None] * 4
                    for d in dots:
                        # Pick the prev id that isn't used yet and is closest.
                        best_i, best_dist = None, float("inf")
                        for j, p in enumerate(prev):
                            if j in used:
                                continue
                            dist = (p[0] - d[0]) ** 2 + (p[1] - d[1]) ** 2
                            if dist < best_dist:
                                best_dist, best_i = dist, j
                        used.add(best_i)
                        matched[best_i] = d
                    for j in range(4):
                        if matched[j] is None:
                            matched[j] = prev[j]
                    out.append(matched)
                    prev = matched
                    continue
                else:
                    out.append(list(prev))
                    continue
        if prev is None:
            ordered = sorted(dots, key=lambda d: (d[0], d[1]))
        else:
            best_perm, best_d = None, float("inf")
            for perm in permutations(range(4)):
                d = sum(
                    ((dots[perm[i]][0] - prev[i][0]) ** 2 +
                     (dots[perm[i]][1] - prev[i][1]) ** 2)
                    for i in range(4)
                )
                if d < best_d:
                    best_d, best_perm = d, perm
            ordered = [dots[p] for p in best_perm]
        out.append(ordered)
        prev = ordered
    return out


def encode(frames_dir: Path, out_path: Path, fps: int = 60) -> None:
    subprocess.run(
        [FFMPEG, "-y", "-loglevel", "error",
         "-framerate", str(fps),
         "-i", str(frames_dir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", "-preset", "medium",
         "-movflags", "+faststart",
         str(out_path)],
        check=True,
    )


def main() -> None:
    # ---- 1) bars in displacement.mp4 ------------------------------------
    disp_frames_dir = WORK / "disp_frames"
    disp_frames = extract_frames(ROOT / "displacement.mp4", disp_frames_dir)
    print(f"displacement: {len(disp_frames)} frames")

    bars = []
    axis_y = 0
    chosen = None
    for f in disp_frames[60::15]:
        b, ay = detect_bars(f)
        if len(b) == 4:
            bars, axis_y, chosen = b, ay, f
            break
    if not bars:
        best = []
        for f in disp_frames:
            b, ay = detect_bars(f)
            if len(b) > len(best):
                best, axis_y, chosen = b, ay, f
        bars = best
    print(f"bars detected ({len(bars)}) in {chosen.name}, axis_y={axis_y}:")
    for cx, cy, rgb, y0, y1 in bars:
        print(f"  cx={cx:6.1f} y[{y0:3d}..{y1:3d}] rgb={tuple(int(v) for v in rgb)}")

    bar_x = [b[0] for b in bars]
    bar_colors_01 = [tuple(float(v) / 255 for v in b[2]) for b in bars]

    # ---- 2) labels ------------------------------------------------------
    s_labels = [render_label(rf"$s_{i+1}$", c, fontsize=22)
                for i, c in enumerate(bar_colors_01)]
    d_labels = [render_label(rf"$d_{i+1}$", c, fontsize=15, stroke=3.0)
                for i, c in enumerate(bar_colors_01)]
    # Save for debugging.
    for i, (sl, dl) in enumerate(zip(s_labels, d_labels)):
        sl.save(WORK / f"label_s{i+1}.png")
        dl.save(WORK / f"label_d{i+1}.png")

    # ---- 3) dots in top.mp4 --------------------------------------------
    top_frames_dir = WORK / "top_frames"
    top_frames = extract_frames(ROOT / "top.mp4", top_frames_dir)
    print(f"top: {len(top_frames)} frames")

    detections = [detect_dots(f) for f in top_frames]
    counts = [len(d) for d in detections]
    print(f"dots/frame: min={min(counts)} max={max(counts)} median={int(np.median(counts))}")

    tracked = track(detections)

    top_out_dir = WORK / "top_labeled"
    top_out_dir.mkdir(parents=True, exist_ok=True)
    DOT_RADIUS = 6     # approximate visible radius of the site dot, px
    CLEARANCE = 6      # gap between dot and label edge, px
    for f, dots in zip(top_frames, tracked):
        img = Image.open(f).convert("RGBA")
        W, H = img.size
        cx_disk, cy_disk = W / 2, H / 2
        for i, (dx, dy) in enumerate(dots):
            label = s_labels[i]
            lw, lh = label.size
            half_w, half_h = lw / 2, lh / 2
            # Unit vector pointing radially outward from the disk centre.
            vx, vy = dx - cx_disk, dy - cy_disk
            n = (vx * vx + vy * vy) ** 0.5
            if n < 1e-3:
                ux, uy = 1.0, 0.0
            else:
                ux, uy = vx / n, vy / n
            # Distance from label centre to its bounding-box edge along
            # the outward direction. Adding the dot's radius and a small
            # clearance puts the *near edge* of the label just past the
            # dot, so the dot stays fully visible.
            edge = min(half_w / max(abs(ux), 1e-3),
                       half_h / max(abs(uy), 1e-3))
            offset = edge + DOT_RADIUS + CLEARANCE
            tx = int(dx + ux * offset - half_w)
            ty = int(dy + uy * offset - half_h)
            tx = max(0, min(W - lw, tx))
            ty = max(0, min(H - lh, ty))
            img.paste(label, (tx, ty), label)
        img.convert("RGB").save(top_out_dir / f.name)

    # ---- 4) bars in displacement.mp4 (dynamic per-frame placement) -----
    disp_out_dir = WORK / "disp_labeled"
    disp_out_dir.mkdir(parents=True, exist_ok=True)
    # Pre-detect per-frame bar geometry so labels follow the changing bar
    # heights. We match each detection back to the canonical (sorted-by-x)
    # bar order so identities stay consistent.
    fallback = [(bx, axis_y, axis_y) for bx in bar_x]  # (cx, y0, y1)
    for f in disp_frames:
        b_per_frame, axy = detect_bars(f)
        # Assign each detection to its closest x in bar_x.
        per_id = list(fallback)
        used = set()
        for cx, cy, rgb, y0, y1 in b_per_frame:
            best_i, best_d = None, float("inf")
            for j, bx in enumerate(bar_x):
                if j in used:
                    continue
                d = abs(cx - bx)
                if d < best_d:
                    best_d, best_i = d, j
            if best_i is not None and best_d < 30:
                used.add(best_i)
                per_id[best_i] = (cx, y0, y1)

        img = Image.open(f).convert("RGBA")
        W, H = img.size
        for i, (cx, y0, y1) in enumerate(per_id):
            label = d_labels[i]
            lw, lh = label.size
            # Positive bar -> above its top; negative bar -> below its bottom.
            if (y0 + y1) / 2 < axy:
                ty = max(0, y0 - lh - 2)
            else:
                ty = min(H - lh, y1 + 2)
            tx = int(cx - lw / 2)
            tx = max(0, min(W - lw, tx))
            img.paste(label, (tx, ty), label)
        img.convert("RGB").save(disp_out_dir / f.name)

    # ---- 5) encode ------------------------------------------------------
    print("encoding top_labeled.mp4 ...")
    encode(top_out_dir, ROOT / "top_labeled.mp4", fps=60)
    print("encoding displacement_labeled.mp4 ...")
    encode(disp_out_dir, ROOT / "displacement_labeled.mp4", fps=60)
    print("done")


if __name__ == "__main__":
    main()
