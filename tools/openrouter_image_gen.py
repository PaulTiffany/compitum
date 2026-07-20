"""Generate text-free scene bitmaps via OpenRouter image models.

Companion to openrouter_media_lab.py. Reuses the same .env loading and never
prints the API key. Requests image output (modalities=["image","text"]) and
writes each returned image to disk. Intended for generating the *background*
layer of a hybrid figure -- the deterministic SVG overlay carries all text and
claims, so prompts here must explicitly forbid embedded words/numbers.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Optional
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_env(path: Path) -> None:
    """Load KEY=value lines from a local .env without overriding the environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


DEFAULT_MODEL = "google/gemini-2.5-flash-image"


def post(payload: dict) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set in the environment or .env.")
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://compitum.space/"),
            "X-Title": os.environ.get("OPENROUTER_APP_NAME", "Compitum Media Lab"),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"OpenRouter HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}"
        ) from exc


def extract_images(response: dict) -> list[bytes]:
    out: list[bytes] = []
    choices = response.get("choices") or []
    for choice in choices:
        msg = (choice or {}).get("message") or {}
        for img in msg.get("images") or []:
            url = (img.get("image_url") or {}).get("url", "") if isinstance(img, dict) else ""
            if url.startswith("data:") and ";base64," in url:
                out.append(base64.b64decode(url.split(";base64,", 1)[1]))
        # Some models nest images inside content blocks.
        content = msg.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict):
                    url = (
                        (blk.get("image_url") or {}).get("url", "")
                        if blk.get("type") == "image_url"
                        else ""
                    )
                    if url.startswith("data:") and ";base64," in url:
                        out.append(base64.b64decode(url.split(";base64,", 1)[1]))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file", default=".env")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True, help="Output PNG path (first image).")
    args = p.parse_args(argv or sys.argv[1:])

    load_env(Path(args.env_file))
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "modalities": ["image", "text"],
    }
    response = post(payload)
    images = extract_images(response)
    if not images:
        # Surface a trimmed response so failures are debuggable without the key.
        text = json.dumps(response, indent=2)[:1200]
        raise SystemExit(f"No image returned. Response head:\n{text}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(images[0])
    print(f"Wrote {out} ({len(images[0]) / 1024:.0f} KB); {len(images)} image(s) total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
