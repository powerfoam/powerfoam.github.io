"""Render 'Reflection' and 'Refraction' caption PNGs with a soft white glow.

The glow imitates the example screenshot: bold white text on a transparent
background, with a wide gaussian halo of the same colour underneath so it
reads cleanly on a colourful video frame.

The PNGs are sized so the *text* portion fills 40 % of the 1920 px video
width, leaving padding around it for the blur to bleed without being clipped.
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path

VIDEO_W = 1920
TARGET_TEXT_W = int(VIDEO_W * 0.40)
# Match the page typography: index.css declares `font-family: 'Jost',
# sans-serif`. We bundled Jost (Google Fonts) under static/fonts/.
FONT_PATH = "static/fonts/Jost-Black.ttf"
GLOW_RADIUS = 22
GLOW_PASSES = 3
PADDING = 160

def fit_font(text: str, target_w: int, font_path: str) -> ImageFont.FreeTypeFont:
    """Binary search for a font size whose rendered width matches `target_w`."""
    lo, hi = 10, 1000
    chosen = ImageFont.truetype(font_path, lo)
    img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(img)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        bbox = draw.textbbox((0, 0), text, font=f)
        w = bbox[2] - bbox[0]
        if w < target_w:
            chosen = f
            lo = mid + 1
        else:
            hi = mid - 1
    return chosen


def render_caption(text: str, out_path: Path) -> None:
    font = fit_font(text, TARGET_TEXT_W, FONT_PATH)

    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = measure.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    canvas_w = text_w + PADDING * 2
    canvas_h = text_h + PADDING * 2
    img = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    tx = PADDING - bbox[0]
    ty = PADDING - bbox[1]

    glow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.text((tx, ty), text, font=font, fill=(0, 0, 0, 255))
    for _ in range(GLOW_PASSES):
        glow = glow.filter(ImageFilter.GaussianBlur(radius=GLOW_RADIUS))

    boosted = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    for _ in range(3):
        boosted = Image.alpha_composite(boosted, glow)

    img = Image.alpha_composite(img, boosted)
    draw = ImageDraw.Draw(img)
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))

    img.save(out_path)
    print(f"{out_path.name}: font={font.size}px, text={text_w}x{text_h}, "
          f"canvas={canvas_w}x{canvas_h}")


if __name__ == "__main__":
    out_dir = Path("video/_overlay")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_caption("Reflection", out_dir / "reflection.png")
    render_caption("Refraction", out_dir / "refraction.png")
