#!/usr/bin/env bash
# Render the one-pager to PDF.
#
# Headless Chrome is used rather than a PDF library because it applies the
# page's own @media print rules, so the PDF and the browser's Cmd-P output are
# the same document. Anything that reflows the layout will show up here first.
set -euo pipefail
cd "$(dirname "$0")"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SRC="onepager.html"
OUT="Pitch-Shape-One-Pager.pdf"

if [ ! -x "$CHROME" ]; then
  echo "Chrome not found at: $CHROME" >&2
  echo "Open $SRC in any browser and use Cmd-P > Save as PDF instead." >&2
  exit 1
fi

"$CHROME" \
  --headless \
  --disable-gpu \
  --no-pdf-header-footer \
  --virtual-time-budget=4000 \
  --print-to-pdf="$PWD/$OUT" \
  "file://$PWD/$SRC" 2>/dev/null

echo "wrote $OUT"

# A one-pager that silently becomes two pages is worse than useless, so fail
# loudly rather than let it ship.
if command -v python3 >/dev/null && [ -x .venv/bin/python ]; then
  .venv/bin/python - "$OUT" <<'PY'
import sys
from pypdf import PdfReader
r = PdfReader(sys.argv[1])
box = r.pages[0].mediabox
print(f"  {len(r.pages)} page(s), {float(box.width)/72:.2f} x {float(box.height)/72:.2f} in")
if len(r.pages) != 1:
    raise SystemExit(f"ERROR: expected 1 page, got {len(r.pages)}")
PY
fi
