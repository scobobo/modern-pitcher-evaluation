"""Wrap paper.html into a fully standalone HTML document.

paper.html is authored for the Artifact publisher, which supplies the
<!doctype>, <head>, and <body> itself. Opened as a local file that wrapper is
missing, so the character encoding falls back to the browser default and the
em-dashes, multiplication signs, and Greek letters render as mojibake.

This produces a self-contained file that opens correctly anywhere, can be
uploaded to any static host, and prints cleanly to PDF.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper.html"
OUTPUT = ROOT / "The-Shape-of-the-Modern-Pitch.html"

# Where the paper is hosted. Open Graph images must be absolute URLs -- a
# relative path renders no preview card at all on LinkedIn, X, or Slack.
SITE_URL = "https://the-shape-of-the-modern-pitch.netlify.app"
# JPEG, and a filename LinkedIn has not seen before: its Open Graph cache is
# sticky, and a URL it once scraped while the image was missing keeps
# returning no card even after the image is fixed.
SOCIAL_CARD = "social-card-v2.jpg"

# A folder that can be dropped straight onto a static host: index.html plus the
# preview image the meta tags point at.
DEPLOY_DIR = ROOT / "site"

# The tab icon, inlined as a data URI rather than shipped as a separate file --
# a second asset is one more thing to forget in a drag-and-drop deploy, and the
# social card has already been lost that way once. The mark is the paper's own
# motif: a ball flight arriving flat.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#11161C"/>'
    '<path d="M5 9C13 11 21 15 26.5 22.5" stroke="#1baf7a" stroke-width="3.2" '
    'fill="none" stroke-linecap="round"/>'
    '<circle cx="26.5" cy="22.5" r="3.4" fill="#1baf7a"/>'
    "</svg>"
)

DESCRIPTION = (
    "A pitch-level study of 7.5 million Statcast pitches (2015-2025) showing that "
    "ball-flight geometry, not velocity or spin rate, drives pitch outcomes - with a "
    "practical framework for evaluating modern pitchers."
)

# Print rules live only in the standalone build: the published artifact is read
# on screen, but a downloadable paper gets printed to PDF, and the sticky
# section rail and hover tooltip are meaningless on paper.
PRINT_CSS = """
    @media print {
      body { background: #fff; }
      .wrap { display: block; max-width: none; padding: 0; }
      .rail, #tip { display: none !important; }
      .sheet { border: 0; padding: 0; }
      figure, .tier, .plain, .caution, .tbl-scroll { break-inside: avoid; }
      h2, h3 { break-after: avoid; }
      a { text-decoration: none; color: inherit; }
      table { font-size: 9pt; }
    }
"""


def main() -> None:
    content = SOURCE.read_text(encoding="utf-8")

    # Lift the <title> out of the fragment and into a real <head>.
    match = re.search(r"<title>(.*?)</title>", content, re.DOTALL)
    title = match.group(1).strip() if match else "The Shape of the Modern Pitch"
    body = re.sub(r"<title>.*?</title>\s*", "", content, count=1, flags=re.DOTALL)

    # Inject the print stylesheet at the end of the existing <style> block so it
    # wins on specificity ties without duplicating a second style element.
    idx = body.rfind("</style>")
    if idx != -1:
        body = body[:idx] + PRINT_CSS + body[idx:]

    favicon = "data:image/svg+xml," + quote(FAVICON_SVG, safe="")

    # Declare the card's real dimensions. Hardcoding them means the tags start
    # lying the moment the render resolution changes.
    card_src = ROOT / "output" / "paper" / "figures" / SOCIAL_CARD
    card_w, card_h = 1200, 630
    if card_src.exists():
        from PIL import Image

        with Image.open(card_src) as _im:
            card_w, card_h = _im.size

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{DESCRIPTION}">
<meta name="author" content="Scott Luntz">
<link rel="icon" href="{favicon}">
<link rel="apple-touch-icon" href="{favicon}">
<meta name="theme-color" content="#11161C">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:type" content="article">
<meta property="og:url" content="{SITE_URL}/">
<meta name="image" property="og:image" content="{SITE_URL}/{SOCIAL_CARD}">
<meta property="og:image:width" content="{card_w}">
<meta property="og:image:height" content="{card_h}">
<meta property="og:image:alt" content="Chart: below about 80 pitches, a pitcher's ball flight predicts his next season better than his own results do.">
<meta property="og:site_name" content="The Shape of the Modern Pitch">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{DESCRIPTION}">
<meta name="twitter:image" content="{SITE_URL}/{SOCIAL_CARD}">
<style>
  html {{ -webkit-text-size-adjust: 100%; }}
  body {{ margin: 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""

    OUTPUT.write_text(html, encoding="utf-8")
    kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.name} ({kb:.0f} KB, self-contained)")

    # Assemble the drag-and-drop deploy folder.
    import shutil

    DEPLOY_DIR.mkdir(exist_ok=True)
    (DEPLOY_DIR / "index.html").write_text(html, encoding="utf-8")

    if card_src.exists():
        shutil.copy(card_src, DEPLOY_DIR / SOCIAL_CARD)
        print(f"wrote site/index.html + site/{SOCIAL_CARD} -> drag `site/` onto your host")
    else:
        print(f"WARNING: {card_src} missing - run src/paper_figures.py first; "
              "the preview card will 404 and no link card will render")


if __name__ == "__main__":
    main()
