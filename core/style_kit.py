"""Shared Pillow drawing kit for Lingxi glassmorphism images.

All public helpers take LOGICAL coordinates on a 1080-wide canvas; rendering
happens at 2x (supersampled) and is downscaled at the end for crisp
anti-aliasing. Alpha colors are blended through transparent layers because a
direct ``ImageDraw`` on the base canvas would replace pixel alpha instead of
compositing.
"""

from __future__ import annotations

import math
from functools import lru_cache
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

S = 2  # supersample factor

_FONT_FILES = {
    "sans": [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ],
    "serif": [
        Path(r"C:\Windows\Fonts\NotoSerifSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ],
    "num": [
        Path(r"C:\Windows\Fonts\Bahnschrift.ttf"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ],
}


def c(hexstr: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Convert a ``#RRGGBB`` string to an RGBA tuple."""
    hexstr = str(hexstr).lstrip("#")
    return (
        int(hexstr[0:2], 16),
        int(hexstr[2:4], 16),
        int(hexstr[4:6], 16),
        alpha,
    )


def mix(col1, col2, t: float):
    """Linearly interpolate the RGB channels of two colors."""
    return tuple(int(a + (b - a) * t) for a, b in zip(col1[:3], col2[:3]))


def mixa(col1, col2, t: float):
    """Linearly interpolate all four channels of two RGBA colors."""
    return tuple(int(a + (b - a) * t) for a, b in zip(col1, col2))


@lru_cache(maxsize=512)
def font(size: int, weight: int = 400, family: str = "sans"):
    """Load a cached font at logical ``size`` with the requested weight.

    Variable fonts (Noto SC / Bahnschrift) honor the ``weight`` axis; static
    fallback fonts silently ignore it.
    """
    resolved = None
    for path in _FONT_FILES.get(family, _FONT_FILES["sans"]):
        if path.is_file():
            resolved = path
            break
    if resolved is None:
        loaded = ImageFont.load_default(int(size * S))
    else:
        loaded = ImageFont.truetype(str(resolved), int(size * S))
    try:
        loaded.set_variation_by_axes([weight])
    except Exception:
        pass
    return loaded


def fsize(fnt) -> float:
    """Logical size of a kit font (stored at 2x)."""
    return fnt.size / S


def _lerp_stops(stops, t: float):
    for i in range(len(stops) - 1):
        p0, col0 = stops[i]
        p1, col1 = stops[i + 1]
        if t <= p1 or i == len(stops) - 2:
            span = max(1e-6, p1 - p0)
            local = min(1.0, max(0.0, (t - p0) / span))
            return mixa(col0, col1, local)
    return stops[-1][1]


class Canvas:
    """A 1080-wide logical drawing surface with glass-style primitives."""

    def __init__(self, height: int, bg: str = "#FAFAF7"):
        self.w = 1080
        self.h = height
        self.img = Image.new("RGBA", (self.w * S, height * S), c(bg))
        self.d = ImageDraw.Draw(self.img)

    # -- text ----------------------------------------------------------
    @staticmethod
    def _soft(col) -> bool:
        return isinstance(col, (tuple, list)) and len(col) == 4 and col[3] < 255

    def _layered(self, fn):
        """Draw via a transparent layer so alpha colors blend correctly."""
        layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        fn(ImageDraw.Draw(layer))
        self.img.alpha_composite(layer)

    def text(self, x, y, s, f, fill, anchor="la"):
        if self._soft(fill):
            self._layered(
                lambda d: d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)
            )
            return
        self.d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)

    def tlen(self, s, f) -> float:
        """Logical pixel width of ``s`` rendered with ``f``."""
        return self.d.textlength(s, font=f) / S

    def spaced(self, x, y, s, f, fill, tracking=5, anchor="la") -> float:
        """Draw letter-spaced text; returns its total logical width."""
        widths = [self.d.textlength(ch, font=f) for ch in s]
        total = (sum(widths) + tracking * S * max(0, len(s) - 1)) / S
        if anchor[0] == "m":
            cx = x - total / 2
        elif anchor[0] == "r":
            cx = x - total
        else:
            cx = x
        for ch, w in zip(s, widths):
            self.d.text((cx * S, y * S), ch, font=f, fill=fill, anchor="l" + anchor[1])
            cx += w / S + tracking
        return total

    def wrap(self, s, f, maxw) -> list[str]:
        """Character-based wrapping suited to CJK text."""
        lines = []
        for para in str(s).splitlines() or [""]:
            line = ""
            for ch in para:
                if line and self.tlen(line + ch, f) > maxw:
                    lines.append(line)
                    line = ch
                else:
                    line += ch
            if line:
                lines.append(line)
        return lines or [""]

    # -- shapes ---------------------------------------------------------
    def line(self, x0, y0, x1, y1, fill, w=1):
        if self._soft(fill):
            self._layered(
                lambda d: d.line(
                    (x0 * S, y0 * S, x1 * S, y1 * S),
                    fill=fill,
                    width=max(1, int(w * S)),
                )
            )
            return
        self.d.line(
            (x0 * S, y0 * S, x1 * S, y1 * S), fill=fill, width=max(1, int(w * S))
        )

    def hline(self, x0, x1, y, fill, w=1):
        self.line(x0, y, x1, y, fill, w)

    def vline(self, x, y0, y1, fill, w=1):
        self.line(x, y0, x, y1, fill, w)

    def rrect(self, box, radius, fill=None, outline=None, width=1):
        if self._soft(fill) or self._soft(outline):
            self._layered(
                lambda d: d.rounded_rectangle(
                    [v * S for v in box],
                    radius * S,
                    fill=fill,
                    outline=outline,
                    width=max(1, int(width * S)),
                )
            )
            return
        self.d.rounded_rectangle(
            [v * S for v in box],
            radius * S,
            fill=fill,
            outline=outline,
            width=max(1, int(width * S)),
        )

    def ellipse(self, box, fill=None, outline=None, width=1):
        if self._soft(fill) or self._soft(outline):
            self._layered(
                lambda d: d.ellipse(
                    [v * S for v in box],
                    fill=fill,
                    outline=outline,
                    width=max(1, int(width * S)),
                )
            )
            return
        self.d.ellipse(
            [v * S for v in box],
            fill=fill,
            outline=outline,
            width=max(1, int(width * S)),
        )

    def dot(self, cx, cy, r, fill):
        self.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)

    def ring(self, cx, cy, r, color, w=2):
        self.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=w)

    def polygon(self, pts, fill):
        if self._soft(fill):
            self._layered(
                lambda d: d.polygon([(x * S, y * S) for x, y in pts], fill=fill)
            )
            return
        self.d.polygon([(x * S, y * S) for x, y in pts], fill=fill)

    def arc(self, box, a0, a1, color, w):
        if self._soft(color):
            self._layered(
                lambda d: d.arc(
                    [v * S for v in box],
                    a0,
                    a1,
                    fill=color,
                    width=max(1, int(w * S)),
                )
            )
            return
        self.d.arc([v * S for v in box], a0, a1, fill=color, width=max(1, int(w * S)))

    def arc_gradient(self, box, a0, a1, col0, col1, w, segs=28):
        """Draw an arc whose stroke color blends from ``col0`` to ``col1``."""
        for i in range(segs):
            s0 = a0 + (a1 - a0) * i / segs
            s1 = a0 + (a1 - a0) * (i + 1) / segs + 0.8
            col = mix(col0, col1, i / max(1, segs - 1))
            self.arc(box, s0, s1, col, w)

    def arc_caps(self, box, a0, a1, color, w):
        """Draw an arc with round end caps at both angles."""
        self.arc(box, a0, a1, color, w)
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        rx = (box[2] - box[0]) / 2 - w / 2
        for ang in (a0, a1):
            rad = math.radians(ang)
            self.dot(cx + rx * math.cos(rad), cy + rx * math.sin(rad), w / 2, color)

    # -- backgrounds ------------------------------------------------------
    def bg_gradient(self, stops):
        """Fill the whole canvas with a vertical multi-stop gradient."""
        h = self.h * S
        strip = Image.new("RGBA", (1, h))
        px = strip.load()
        for y in range(h):
            px[0, y] = _lerp_stops(stops, y / max(1, h - 1))
        self.img.paste(strip.resize((self.w * S, h), Image.BILINEAR), (0, 0))

    def glow(self, cx, cy, r, color, alpha=70, steps=16):
        """Paint a soft radial glow blob behind glass panels."""
        layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        col = color[:3] if isinstance(color, (tuple, list)) else c(color)[:3]
        for i in range(steps, 0, -1):
            t = i / steps
            rr = r * t
            a = int(alpha * (1 - t) ** 1.7)
            d.ellipse(
                ((cx - rr) * S, (cy - rr) * S, (cx + rr) * S, (cy + rr) * S),
                fill=(*col, a),
            )
        layer = layer.filter(ImageFilter.GaussianBlur(max(3, r * S // 24)))
        self.img.alpha_composite(layer)

    def shadow(self, box, radius=28, blur=20, dy=10, color=(30, 34, 48), alpha=60):
        """Blurred drop shadow under a rounded panel."""
        x0, y0, x1, y1 = [v * S for v in box]
        m = blur * S * 3
        cx0, cy0 = max(0, int(x0 - m)), max(0, int(y0 + dy * S - m))
        cx1 = min(self.img.width, int(x1 + m))
        cy1 = min(self.img.height, int(y1 + dy * S + m))
        if cx1 <= cx0 or cy1 <= cy0:
            return
        layer = Image.new("RGBA", (cx1 - cx0, cy1 - cy0), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle(
            (x0 - cx0, y0 + dy * S - cy0, x1 - cx0, y1 + dy * S - cy0),
            radius * S,
            fill=(*color, alpha),
        )
        self.img.alpha_composite(
            layer.filter(ImageFilter.GaussianBlur(blur * S)), (cx0, cy0)
        )

    def glass(
        self,
        box,
        radius=28,
        tint=(255, 255, 255),
        alpha=100,
        blur=16,
        outline=None,
        owidth=1,
    ):
        """Frosted panel: blur the background under ``box`` and overlay tint."""
        x0, y0, x1, y1 = [int(v * S) for v in box]
        region = self.img.crop((x0, y0, x1, y1)).filter(
            ImageFilter.GaussianBlur(blur * S // 2)
        )
        layer = Image.new("RGBA", region.size, (*tint, alpha))
        region = Image.alpha_composite(region, layer)
        mask = Image.new("L", region.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, region.width - 1, region.height - 1), radius * S, fill=255
        )
        self.img.paste(region, (x0, y0), mask)
        if outline:
            self.rrect(box, radius, outline=outline, width=owidth)

    # -- decorations -------------------------------------------------------
    def avatar(self, path, cx, cy, dsize, ring, ring_w=3):
        """Composite the plugin ``logo.png`` as a ringed circular avatar."""
        size = int(dsize * S)
        try:
            src = Image.open(path).convert("RGB")
        except Exception:
            return
        src = ImageOps.fit(
            src, (size, size), Image.Resampling.LANCZOS, centering=(0.5, 0.42)
        )
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
        pad = max(2, int(ring_w * S))
        out = Image.new("RGBA", (size + pad * 2, size + pad * 2), (0, 0, 0, 0))
        ImageDraw.Draw(out).ellipse((0, 0, out.width - 1, out.height - 1), fill=ring)
        out.paste(src, (pad, pad), mask)
        self.img.alpha_composite(
            out, (int(cx * S - out.width / 2), int(cy * S - out.height / 2))
        )

    def pill(self, x, y, text, f, fg, bg, padx=18, pady=8, anchor="la", radius=None):
        """Rounded tag pill; ``anchor`` follows the same convention as text."""
        tw = self.tlen(text, f)
        th = fsize(f)
        w, h = tw + padx * 2, th + pady * 2
        if anchor[0] == "m":
            box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        elif anchor[0] == "r":
            box = (x - w, y, x, y + h)
        else:
            box = (x, y, x + w, y + h)
        self.rrect(box, radius if radius is not None else h / 2, fill=bg)
        self.text(
            (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, text, f, fg, anchor="mm"
        )
        return box

    def star4(self, cx, cy, r, fill, ratio=0.32):
        """Four-point sparkle star used as decoration."""
        pts = []
        for i in range(8):
            a = i * math.pi / 4 - math.pi / 2
            rad = r if i % 2 == 0 else r * ratio
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        self.polygon(pts, fill)

    def heart(self, cx, cy, s, fill):
        """Small heart glyph built from two circles and a triangle."""
        lw = int(s * S * 1.2)
        layer = Image.new("RGBA", (lw, lw), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        ccx, ccy = lw / 2, lw / 2
        u = s * S
        r = u * 0.26
        off = u * 0.24
        up = u * 0.14
        d.ellipse((ccx - off - r, ccy - up - r, ccx - off + r, ccy - up + r), fill=fill)
        d.ellipse((ccx + off - r, ccy - up - r, ccx + off + r, ccy - up + r), fill=fill)
        d.polygon(
            [
                (ccx - u * 0.475, ccy - u * 0.02),
                (ccx + u * 0.475, ccy - u * 0.02),
                (ccx, ccy + u * 0.46),
            ],
            fill=fill,
        )
        self.img.alpha_composite(layer, (int(cx * S - lw / 2), int(cy * S - lw / 2)))

    def bookmark(self, cx, cy, w, h, fill):
        """Bookmark ribbon glyph."""
        self.polygon(
            [
                (cx - w / 2, cy - h / 2),
                (cx + w / 2, cy - h / 2),
                (cx + w / 2, cy + h / 2),
                (cx, cy + h / 2 - w * 0.28),
                (cx - w / 2, cy + h / 2),
            ],
            fill,
        )

    # -- output -------------------------------------------------------------
    def finish(self, bottom: int) -> bytes:
        """Crop to ``bottom``, downscale to 1080 width, and encode PNG."""
        img = self.img.crop((0, 0, self.w * S, int(bottom * S)))
        img = img.resize((self.w, int(bottom)), Image.LANCZOS).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
