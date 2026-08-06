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

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper.html"
OUTPUT = ROOT / "The-Shape-of-the-Modern-Pitch.html"

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

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{DESCRIPTION}">
<meta name="author" content="Scott Luntz">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{DESCRIPTION}">
<meta property="og:type" content="article">
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
    print(f"wrote {OUTPUT.name} ({kb:.0f} KB, self-contained, no external requests)")


if __name__ == "__main__":
    main()
