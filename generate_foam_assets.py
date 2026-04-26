"""Regenerate the foam-bubble brand assets used by the webpage.

Outputs:

  static/images/foam-text-pattern.png
      Wide bubble texture used as `background-image` on h2 section
      headers (via background-clip: text). Voronoi cells filled with a
      purple-blue -> pink-red gradient that matches the title wordmark.

  static/images/foam-bubbles-bg.svg
      A handful of large, very faint foam blobs placed at fixed positions
      and pinned behind the page content as a subtle brand accent.

Run from the repo root:

    python3 generate_foam_assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.spatial import Voronoi


OUT_DIR = Path(__file__).resolve().parent / "static" / "images"


# ---------------------------------------------------------------------------
# 1. Tileable bubble texture for section headers
# ---------------------------------------------------------------------------

def render_text_pattern(out_path: Path) -> None:
    tex_w, tex_h = 1600, 220
    num_sites = 380

    # Slightly punchier than the title image so the headings still pop
    # when sized down to fit a single h2.title.is-3 line.
    left_col = np.array([138, 110, 220])   # purple-blue
    right_col = np.array([226,  90, 150])  # pink-red

    rng = np.random.default_rng(42)

    # Lay out sites with extra padding around the visible area so border
    # cells have finite vertices.
    margin_x, margin_y = 80, 80
    xs = rng.uniform(-margin_x, tex_w + margin_x, num_sites)
    ys = rng.uniform(-margin_y, tex_h + margin_y, num_sites)
    sites = np.stack([xs, ys], axis=1)

    # Pin "infinity" sites at far-out corners so all interior Voronoi
    # regions are bounded.
    sites = np.vstack([
        sites,
        [[-3 * tex_w, -3 * tex_h],
         [ 4 * tex_w, -3 * tex_h],
         [-3 * tex_w,  4 * tex_h],
         [ 4 * tex_w,  4 * tex_h]],
    ])
    vor = Voronoi(sites)

    img = Image.new("RGBA", (tex_w, tex_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    for site_idx, region_idx in enumerate(vor.point_region):
        if site_idx >= num_sites:
            continue
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        poly = [tuple(vor.vertices[i]) for i in region]

        cx = sites[site_idx, 0]
        # Horizontal gradient mix.
        t = float(np.clip(cx / tex_w, 0.0, 1.0))
        rgb = (1 - t) * left_col + t * right_col

        # Per-cell lightness jitter for organic feel.
        jitter = rng.uniform(-12, 12)
        color = tuple(int(c) for c in np.clip(rgb + jitter, 0, 255).astype(int))

        draw.polygon(poly, fill=color, outline=(255, 255, 255, 255))

    # Soften edges so the cells look organic, not CGI.
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, optimize=True)
    print(f"wrote {out_path}  ({tex_w}x{tex_h})")


# ---------------------------------------------------------------------------
# 2. Decorative background SVG (sparse, low-opacity foam blobs)
# ---------------------------------------------------------------------------

def render_background_svg(out_path: Path) -> None:
    svg_w, svg_h = 1600, 2400
    bubbles = [
        # x, y, r, color
        ( 180,  240, 220, "#a98ce3"),
        (1380,  480, 280, "#e8a3c1"),
        ( 320, 1100, 320, "#a98ce3"),
        (1320, 1500, 240, "#e8a3c1"),
        ( 220, 1900, 200, "#a98ce3"),
        (1450, 2100, 260, "#e8a3c1"),
    ]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" '
        'preserveAspectRatio="xMidYMid slice" aria-hidden="true">',
        '<defs>',
    ]
    for i, (_cx, _cy, _r, color) in enumerate(bubbles):
        parts.append(
            f'<radialGradient id="g{i}" cx="50%" cy="50%" r="50%">'
            f'<stop offset="0%" stop-color="{color}" stop-opacity="0.42"/>'
            f'<stop offset="60%" stop-color="{color}" stop-opacity="0.18"/>'
            f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
            '</radialGradient>'
        )
    parts.append('</defs>')
    for i, (cx, cy, r, _color) in enumerate(bubbles):
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#g{i})"/>')
    parts.append('</svg>')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    print(f"wrote {out_path}")


def main() -> None:
    render_text_pattern(OUT_DIR / "foam-text-pattern.png")
    render_background_svg(OUT_DIR / "foam-bubbles-bg.svg")


if __name__ == "__main__":
    main()
