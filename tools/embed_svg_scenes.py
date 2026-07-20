#!/usr/bin/env python3
"""Make hybrid SVGs self-contained for web/GitHub embedding.

The reviewed "hybrid" figures layer deterministic SVG labels over a generated
bitmap scene referenced by a *relative external* path
(``<image href="..._scene_v1.png">``). That renders when the .svg is opened as a
top-level document, but NOT when it is embedded via ``<img src>`` or GitHub's
image pipeline: browsers run embedded SVGs in secure-static mode and block
external resource fetches, so the scene layer disappears.

Fix: downscale + JPEG-compress each scene and inline it as a base64 data URI, so
each SVG carries its own pixels and renders identically everywhere. Idempotent:
SVGs already holding a data URI are skipped.
"""

from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path

from PIL import Image

REVIEWED = Path(__file__).resolve().parent.parent / "media" / "reviewed"
MAX_EDGE = 1800
JPEG_QUALITY = 82

HREF_RE = re.compile(r'href="([^"]+_scene_v1\.png)"')


def discover() -> list[str]:
    """Every *_hybrid_v1.svg in media/reviewed (idempotent: embedded ones skip)."""
    return sorted(p.name for p in REVIEWED.glob("*_hybrid_v1.svg"))


def encode_scene(png_path: Path) -> str:
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, MAX_EDGE / max(w, h))
    if scale < 1.0:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def main() -> int:
    for name in discover():
        svg_path = REVIEWED / name
        text = svg_path.read_text(encoding="utf-8")
        if "data:image/jpeg;base64," in text:
            print(f"skip  {name} (already self-contained)")
            continue
        m = HREF_RE.search(text)
        if not m:
            print(f"WARN  {name}: no scene href found", file=sys.stderr)
            continue
        png_path = REVIEWED / m.group(1)
        if not png_path.exists():
            print(f"WARN  {name}: missing scene {png_path.name}", file=sys.stderr)
            continue
        data_uri = encode_scene(png_path)
        new_text = HREF_RE.sub(f'href="{data_uri}"', text, count=1)
        svg_path.write_text(new_text, encoding="utf-8")
        kb = len(new_text.encode("utf-8")) / 1024
        print(f"embed {name}: {png_path.name} -> {kb:.0f} KB self-contained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
