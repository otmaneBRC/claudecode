#!/usr/bin/env python3
"""Download the Higgsfield-generated images used by index.html, compress them
to web-sized WebP, and rewrite index.html to use the local copies.

    pip install pillow
    python3 tools/localize_images.py            # download + compress + rewrite
    python3 tools/localize_images.py --dry-run  # just list what would happen

The originals are 4-5 MB PNGs (~2048px); the output is ~1600px WebP (~150-300 KB).
"""
import argparse, io, re, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "index.html"
OUT = ROOT / "assets" / "img"
URL_RE = re.compile(r"https://d8j0ntlcm91z4\.cloudfront\.net/[^\s\"')]+\.png")
MAX_EDGE, QUALITY = 1600, 80

# job-id fragment -> readable name
NAMES = {
    "d7e4bc1f": "hero", "8f526db9": "crossbody-bag", "1f1b412e": "messenger-bag",
    "e3362094": "cta-background", "70e611db": "card-holder", "070050d7": "bifold-wallet",
    "a269e85a": "pattern-sheets", "4f82a8b7": "everyday-carry", "f08a68af": "tools",
    "dedd78da": "leather-swatches", "ce724aeb": "hands-wallet", "1c81dec0": "stitching",
    "6d237d3e": "scrap-leather", "798f7fa2": "market-stall",
}

def name_for(url):
    for k, v in NAMES.items():
        if k in url:
            return v
    return re.sub(r"\W+", "-", url.rsplit("/", 1)[-1][:-4])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    html = HTML.read_text(encoding="utf-8")
    urls = sorted(set(URL_RE.findall(html)))
    print(f"{len(urls)} remote images found in index.html")
    if args.dry_run:
        for u in urls:
            print(f"  {name_for(u)}.webp  <-  {u}")
        return
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow is required:  pip install pillow")
    OUT.mkdir(parents=True, exist_ok=True)
    for u in urls:
        dest = OUT / f"{name_for(u)}.webp"
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.thumbnail((MAX_EDGE, MAX_EDGE))
        im.save(dest, "WEBP", quality=QUALITY, method=6)
        print(f"  {dest.relative_to(ROOT)}  {len(data)//1024} KB -> {dest.stat().st_size//1024} KB")
        html = html.replace(u, f"assets/img/{dest.name}")
    HTML.write_text(html, encoding="utf-8")
    print("index.html now points at assets/img/*.webp")

if __name__ == "__main__":
    main()
