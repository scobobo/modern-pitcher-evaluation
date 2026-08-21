"""Inject the dataset into the dashboard template to produce a standalone page.

The data is embedded rather than fetched. An Artifact runs under a strict CSP
that blocks requests to other hosts, and a dashboard that needs a live endpoint
is a dashboard that breaks the first time the endpoint moves.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
TEMPLATE = ROOT / "dashboard_template.html"
DATA = ROOT / "output" / "dashboard" / "pitcher_seasons.json"
OUTPUT = ROOT / "dashboard.html"

# Artifacts cap the rendered page at 16 MB; stay well clear of it.
SIZE_LIMIT_MB = 12.0


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA} - run build_dashboard_data.py first")

    template = TEMPLATE.read_text(encoding="utf-8")
    data = DATA.read_text(encoding="utf-8")

    # Validate before embedding: a malformed payload would fail silently in the
    # browser as a blank page.
    parsed = json.loads(data)
    rows, cols = len(parsed["rows"]), len(parsed["columns"])
    if not rows:
        raise SystemExit("dataset is empty")

    # "</script>" anywhere inside the JSON would terminate the host element
    # early. It cannot occur in this data, but the escape is cheap insurance.
    data = data.replace("</", "<\\/")

    if "__DATA__" not in template:
        raise SystemExit("template is missing the __DATA__ placeholder")
    html = template.replace("__DATA__", data)

    OUTPUT.write_text(html, encoding="utf-8")

    # A second copy with a real <head>, for hosting outside the Artifact
    # publisher (which supplies the wrapper itself). Without a declared
    # charset a static host will mis-decode the degree signs and en dashes.
    standalone = ROOT / "site-dashboard" / "index.html"
    standalone.parent.mkdir(exist_ok=True)
    standalone.write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Pitch Shape Explorer</title>\n"
        '<meta name="description" content="Filter 15,014 MLB pitcher-seasons by year, '
        'age, pitch type and volume, scored by a validated pitch-shape model.">\n'
        "</head>\n<body>\n" + html + "\n</body>\n</html>\n",
        encoding="utf-8",
    )
    print(f"wrote site-dashboard/index.html (drag that folder to a static host)")

    mb = OUTPUT.stat().st_size / 1e6
    print(f"wrote {OUTPUT.name} ({mb:.2f} MB, {rows:,} rows x {cols} columns)")
    if mb > SIZE_LIMIT_MB:
        raise SystemExit(f"ERROR: {mb:.1f} MB exceeds the {SIZE_LIMIT_MB} MB budget")


if __name__ == "__main__":
    main()
