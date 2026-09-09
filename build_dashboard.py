"""Inject the dataset into the dashboard template to produce a standalone page.

The data is embedded rather than fetched. An Artifact runs under a strict CSP
that blocks requests to other hosts, and a dashboard that needs a live endpoint
is a dashboard that breaks the first time the endpoint moves.
"""

from __future__ import annotations

import json
import pathlib
import sys

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

    # Regenerate the link-preview card from the same payload the page embeds,
    # so the count on the card cannot drift from the count in the page.
    sys.path.insert(0, str(ROOT / "src"))
    from config import DASHBOARD_URL
    from paper_figures import fig_dashboard_card

    card = fig_dashboard_card(standalone.parent / "dashboard-card.png", (parsed["columns"], parsed["rows"], parsed["meta"]))
    (standalone.parent / "dashboard-card.png").unlink(missing_ok=True)

    title = "Pitch Shape Explorer"
    desc = (f"Filter {rows:,} MLB pitcher-seasons by year, age, pitch type and volume, "
            "each scored against what its ball flight alone predicts.")
    alt = ("Scatter of actual whiff rate against the rate a pitcher's ball flight predicts, "
           "with the pitchers out-throwing their shape marked.")
    img = f"{DASHBOARD_URL.rstrip('/')}/{card.name}"

    # `name="image"` alongside `property="og:image"` is what LinkedIn's own
    # guidance asks for, and its absence was one of three separate causes of a
    # blank card on the paper. Width and height are declared because scrapers
    # that cannot fetch the image inline will still reserve the right shape.
    head = (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f'<meta name="description" content="{desc}">\n'
        f'<meta property="og:title" content="{title}">\n'
        f'<meta property="og:description" content="{desc}">\n'
        '<meta property="og:type" content="website">\n'
        f'<meta property="og:url" content="{DASHBOARD_URL.rstrip("/")}/">\n'
        f'<meta name="image" property="og:image" content="{img}">\n'
        '<meta property="og:image:width" content="2400">\n'
        '<meta property="og:image:height" content="1260">\n'
        f'<meta property="og:image:alt" content="{alt}">\n'
        f'<meta property="og:site_name" content="{title}">\n'
        '<meta name="twitter:card" content="summary_large_image">\n'
        f'<meta name="twitter:title" content="{title}">\n'
        f'<meta name="twitter:description" content="{desc}">\n'
        f'<meta name="twitter:image" content="{img}">\n'
        "</head>\n<body>\n"
    )
    standalone.write_text(head + html + "\n</body>\n</html>\n", encoding="utf-8")
    print(f"wrote site-dashboard/index.html + {card.name} (drag that folder to a static host)")
    print(f"  card URL is {img} -- set DASHBOARD_URL in src/config.py if that is not the host")

    mb = OUTPUT.stat().st_size / 1e6
    print(f"wrote {OUTPUT.name} ({mb:.2f} MB, {rows:,} rows x {cols} columns)")
    if mb > SIZE_LIMIT_MB:
        raise SystemExit(f"ERROR: {mb:.1f} MB exceeds the {SIZE_LIMIT_MB} MB budget")


if __name__ == "__main__":
    main()
