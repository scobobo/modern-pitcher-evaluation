# Handoff — The Shape of the Modern Pitch

Background context for picking this project up cold. Written 26 August 2026.

---

## What this is

A published baseball research project by **Scott Luntz**, plus the tools built
on top of it. The goal behind the work is career-oriented: Scott is a
technologist, not a professional analyst, and this is the portfolio piece
intended to get him taken seriously in baseball analytics. That framing matters
for a lot of the decisions below — credibility is worth more here than
impressiveness, and several choices trade the second for the first.

**Live surfaces**

| What | Where | Status |
| --- | --- | --- |
| Paper (web) | https://the-shape-of-the-modern-pitch.netlify.app/ | live, HTTP 200 |
| Code | https://github.com/scobobo/modern-pitcher-evaluation | pushed, in sync |
| Paper DOI | `10.5281/zenodo.22037431` | resolves |
| Code archive DOI | `10.5281/zenodo.22037409` | — |
| Dashboard (artifact) | `claude.ai/code/artifact/ad575e95-…` | private to Scott |
| One-pager (artifact) | `claude.ai/code/artifact/879985b5-…` | private to Scott |

Local git is level with `origin/main`, nothing unpushed as of this writing.

---

## The finding, in short

Across 7,483,321 Statcast pitches (2015–2025), **pitch shape** — induced
vertical break, horizontal break, and height-adjusted vertical approach angle —
explains **4.5× more** variation in run value than velocity, and **3.4× more**
for whiffs. **Residual spin rate contributes nothing measurable** once velocity
is partialled out, even when granted first claim on the shared variance
(−0.00003 ± 0.00006, t = −0.46).

The more useful half is about measurement error rather than effect size. Shape
metrics have split-half reliability of **r ≈ 0.99**; run value per pitch has
**r = 0.20**. Because shape is measured almost exactly from the first pitch, it
predicts a pitcher's next season better than his own results do **below roughly
80 pitches**. Above that, results win and keep pulling away.

### Boundaries the paper states explicitly

These are load-bearing. The project's credibility rests on them being present,
and past sessions have repeatedly had to resist softening them.

- **Command outranks shape roughly 50:1.** Distance from the middle of the zone
  scores 0.050 in permutation importance; the best physical trait scores
  0.00097. Shape dominates among *pitch-intrinsic* properties only.
- **Shape does not replace results on full seasons.** At 500 pitches, past
  results predict next-season whiff rate at R² 0.362 against shape's 0.188.
- **The study rejected its own starting hypothesis.** It began from Scott's
  proposition that spin had displaced velocity, and the data said no. The claim
  was then narrowed twice more. All three reversals are in §6 of the paper.

---

## Non-goals

- **Injury.** The analysis says nothing about arm health. A commenter drew that
  connection publicly and the reply deliberately declined it. Anything that
  implies velocity chasing causes injury is outside what this data supports.
- **Causal claims.** Everything is observational. Nothing establishes that
  adding two inches of IVB to a given pitcher would improve his results.
- **Refitting the model per dashboard filter.** Considered and rejected; see
  decisions below.
- **Spin efficiency.** Not computable from public Statcast. `spin_axis` is
  inferred from observed movement, not measured from Hawk-Eye's 3D axis.

---

## Repo layout

Analysis pipeline:

```
src/config.py       season windows, pitch groups, constants
src/fetch.py        Statcast pulls, one parquet per season, chunked with retries
src/features.py     approach angles, movement, spin residualisation, outcomes
src/model.py        nested attribution ladder, grouped CV, permutation importance
src/temporal.py     per-season effects and trend tests
src/evaluation.py   reliability, year-over-year stability, next-season forecasting
src/projection.py   shape expectation, gap reversal, projection model
src/leaderboard.py  walk-forward backtest, decile analysis, board builder
src/players.py      birth dates and handedness from MLB StatsAPI, cached
src/plots.py        analysis figures
src/paper_figures.py  publication figures and the social card
```

Entry points:

```
run_analysis.py          attribution ladder (paper §5.1)
run_paper_analysis.py    VAA robustness, pitch-type generality, reliability (§5.3–5.5)
run_sample_size_test.py  the crossover experiment (§5.6)
run_projection.py        regression/progression candidates
run_leaderboard.py       validated leaderboard + walk-forward backtest
build_dashboard_data.py  exports the dashboard dataset as JSON
build_dashboard.py       injects that JSON into the dashboard template
build_standalone.py      standalone HTML edition of the paper
build_docx.py            Word edition
build_pdf.sh             one-pager to PDF via headless Chrome
```

Documents:

```
paper.html              artifact source for the paper (no <head> — publisher supplies it)
onepager.html           one-page executive summary for front-office readers
dashboard_template.html dashboard markup with a __DATA__ placeholder
README.md               repo front door
RESULTS.md              the original findings write-up
```

Generated and gitignored: `dashboard.html`, `site/`, `site-dashboard/`,
`The-Shape-of-the-Modern-Pitch.{html,docx}`, `Pitch-Shape-One-Pager.pdf`,
`data/` (702 MB of parquet), `output/`.

---

## Environment

The machine's system Python is a 3.15 beta with no scientific wheels available,
and `/usr/bin/python3` is blocked by an unaccepted Xcode licence. The project
therefore runs on a **`uv`-managed CPython 3.12 in `.venv/`**, installed with
Scott's approval. Everything is invoked as `.venv/bin/python`.

`pybaseball` needed a workaround: it pulls `cryptography` through a Retrosheet
module the project never touches, and that build fails here. The install order
in the README skips it with `--no-deps`.

Not installed, which shaped several decisions: **Node** (so the dataviz palette
validator could not be run), **LibreOffice / pandoc** (so the docx skill's usual
toolchain was unavailable and `python-docx` was used instead), **poppler** (so
PDFs cannot be rendered to images for visual checking). **Google Chrome is
installed** and is used headless for the one-pager PDF.

---

## Key decisions and why

**Paired fold differences, not point estimates.** Pitch-level run value has an
SD of 0.23 around a mean of zero, so a feature block can "gain" +0.0004 R² by
luck. Because `GroupKFold` is deterministic, each block's gain is differenced
fold by fold to get a standard error. Nothing is called real below t = 2. An
earlier version of the verdict function declared success off a +0.00039 gain
against fold noise of ~0.005; this exists to stop that.

**Every ladder rung scores on identical rows.** `release_extension` is null on
about 2% of pitches, so a per-rung dropna quietly handed later rungs an easier
sample. That inflated velocity's standard error 50× and made the comparison
meaningless before it was fixed.

**Folds grouped by pitcher.** Random pitch-level splits let the model memorise
the arm rather than learn the physics.

**VAA residualised on plate height.** The strongest objection to any shape
finding is that approach angle is mechanically tied to pitch height. Regressing
VAA on height (quadratic, within season and pitch type) and keeping the residual
retains 93% of the shape effect and *improves* the t-statistic.

**The dashboard model is fixed, not refit per filter.** Refitting on whatever
subset the user selected would mean filtering to "lefties aged 24–27 throwing
sweepers" silently reports a model trained on 40 rows. Filtering changes which
rows are shown and nothing else. This is stated on the page.

**Shape expectations clipped to the observed range per pitch type.** A ridge
extrapolates, so a knuckleballer's rare four-seamer drew a 1.1% expected whiff
rate against a league floor near 6%. Only ~9 rows were affected out of 14,800,
but since the boards name the extremes, that one row *was* the headline.

**Named candidates require 300+ pitches**, and the panel says so when too few
qualify. Same reasoning: the extremes of any edge ranking are disproportionately
thin samples.

**Age computed as of 30 June**, baseball's convention. 1 January would shift
about half the population by a year.

**Disclosure is candid rather than minimised.** §10.3 states plainly that AI
assistance produced the pipeline, the statistical testing, the figures and the
draft prose, under Scott's direction. The reasoning: an understated disclosure
that later reads as misleading would cost far more than a candid one, and the
genuinely creditable parts — setting the falsifiability standard, publishing the
disconfirming findings — are his.

**Jimmy Stanley credited briefly, not itemised.** He made 53 prose edits. An
earlier draft listed a specific cross-reference he caught, which read as faint
praise; acknowledgements are conventionally short.

---

## Verified by running

- **Paper numbers.** Full 11-season pipeline over 7,483,321 pitches. Attribution
  ladder in both orderings; spin returned −0.00003 ± 0.00006 (t = −0.46) even
  entering first. Generality across 7 pitch types: shape is the largest block in
  all 14 pitch-type-by-outcome combinations.
- **Physics sanity.** 2025 four-seamers: VAA mean −4.74°, IVB 15.8 in, velo 94.4
  mph, spin 2309 rpm. All match known league values.
- **Residualisation.** corr(velocity, residual spin) = −0.000.
- **Walk-forward backtest**, rerun 6 September 2026 with the rolling window.
  1,413 out-of-sample pitcher-seasons, shape lift **+0.0241** R² over mean
  reversion (was +0.018), positive in **8 of 8** seasons (was 7 of 8),
  corr(edge, what mean reversion missed) = **+0.239, t = +9.2** (was +0.148,
  t = +5.6). Top edge decile beat the bottom by **3.7** points of realised whiff
  rate (was 2.5). The two seasons that had died came back: 2023→2024 lift
  +0.0000 → +0.0226, and 2024→2025 −0.0178 → +0.0104.
- **Signal decay, found and then fixed.** The per-season correlation was
  trending down at −0.024/year (t = −2.42), with 2024 flat and 2025 negative.
  The cause was training on all eleven seasons: the game drifts (league fastball
  velocity is up 1.66 mph since 2015, t = +10.9) so decade-old seasons describe
  pitchers who no longer exist. Training the edge on a rolling **4-season
  window** removes it. The lift trend goes from −0.0034/year to +0.0003/year
  (t = +0.14, p = 0.89), which is flat.

  Era-neutralising the features instead — z-scoring shape within each season —
  was tried first and made things slightly *worse* (recent r 0.193 vs 0.205), so
  the problem is not that the units drifted, it is that the relationship did.

  A 3-season window scored marginally better still but drops to 183 training
  pairs in one year, and picking the top window on the same data used to
  evaluate it overfits the hyperparameter. Four was chosen for stability.
- **No leakage in the dashboard model.** corr(shape expectation, actual) runs
  0.09–0.49 with an sd ratio of 0.27–0.51 across pitch types; in-sample fitting
  would put both near 1.0.
- **End-to-end trace.** Louis Varland's 2026 line recomputed from raw parquet
  independent of the pipeline: 421 pitches, 232 swings, 44 whiffs, 18.97%,
  matching the dashboard exactly along with velo, IVB and adjusted VAA.
- **Per-pitch-type edge validation** (this is the newest finding):

  Those verdicts were **wrong** and were corrected on 6 September 2026. The old
  test scored the shipped `shape_edge`, which is fit on every complete season,
  against pairs drawn from those same seasons, so each pair was judged by a
  model that had seen its own future. Refit walk-forward, using only seasons
  that closed before each board:

  ```
  FF t=+12.6   SI t=+7.5   SL t=+5.5   FC t=+2.4  → validated
  CH t=+1.9    CU t=+1.8                          → promising, short of t=2
  ST n=0                                          → untested, too new to score
  ```

  Two of seven became four of seven by giving the model **fastball-relative
  features** (`SEPARATION_FEATURES` in `src/projection.py`). A secondary pitch
  does not miss bats on its own geometry; it misses bats by differing from the
  fastball hitters are timing. The shape-only feature set had no way to say
  that, which is exactly why it worked on fastballs and nothing else. Adding
  velocity, movement and release separation from each pitcher's primary
  fastball:

  ```
  pitch      shape only   + fastball-relative   named candidates beat naive
  slider     t=+1.3       t=+4.9                +0.68 -> +1.94 pp (55% -> 66%)
  changeup   t=+0.4       t=+4.6                +1.02 -> +2.69 pp (56% -> 65%)
  curveball  t=+0.8       t=+3.8                -0.09 -> +2.25 pp (49% -> 61%)
  cutter     t=+2.6       t=+4.3                +2.60 -> +3.12 pp (64% -> 69%)
  four-seam  t=+10.4      t=+10.2               unchanged, as expected
  ```

  Fastballs are unchanged because a pitch measured against itself has zero
  separation, so those columns are zero for whichever fastball the pitcher leads
  with. Curveball is the one to note: it went from a literal coin flip on named
  candidates to 61%.

  Verdicts are now four states, not two. Collapsing everything below t=2 into
  one orange badge read as "this pitch does not matter", which was both
  discouraging and wrong: a pitch pointing the right way at t=1.9 is nothing
  like one where the effect is flat. `validated` / `promising` / `no signal` /
  `untested`, each with its own wording, and the panel now leads with what is
  proven before listing where the rest stand.

  Only four-seamers and sinkers survive. Cutter, slider and curveball had been
  shipping green "validated" dots on a signal that is not there. The dashboard
  derives the "it holds up on" list from the data now rather than hardcoding it,
  so this cannot drift again.
- **One-pager PDF.** 1 page, 8.50 × 11.00 in, selectable text. `build_pdf.sh`
  fails loudly if it ever becomes two pages.
- **Data currency.** 2026 refreshed through 5 September: 631,402 pitches, 99.7%
  run-value coverage. Dataset is 15,211 pitcher-seasons across 7 pitch types.
- **Board rows trace to raw parquet.** Varland, Misiorowski and Chandler each
  recomputed from `data/statcast_2026.parquet` with no project modules: pitch
  counts, swings, whiff rate, velocity and IVB all match the leaderboard exactly.
- **Gap persistence, measured per pitch type.** Roughly 35-47% of a gap closes
  year over year at the 300-pitch naming threshold, depending on the pitch
  (four-seam 46%, sinker 45%, cutter 47%, changeup 42%, curveball 37%,
  slider 35%). The panel quotes the number for whatever is selected.

  Two traps found here and closed. Pooling every pitch type into one figure
  reported 78% reversion when no individual type is above 47%, which is
  between-type variance masquerading as reversion; the all-pitches number is now
  a weighted average of the per-type ones. And the board-recurrence statistic is
  meaningless for thin pitch types, where naming a top and bottom twenty covers
  forty of about fifty qualifying pitchers, so chance recurrence exceeds the
  observed rate. It is now reported only where the board is genuinely selective,
  which is four-seamers alone: 30% return against 15% by chance.

---

## Unverified assumptions

- **Both DOIs were supplied by Scott and have not been independently confirmed
  to resolve to the right records.** `doi.org/10.5281/zenodo.22037431` returns a
  302, which is expected behaviour for a DOI resolver, but the destination
  content was never checked.
- **The dashboard artifact renders correctly in the artifact viewer.** It was
  verified served over `http://localhost:8899`; the artifact publish reported
  success but the rendered page was not re-inspected there.
- **The LinkedIn preview card was never confirmed rendering in Post Inspector.**
  Three separate causes were found and fixed (missing image, RGBA alpha channel,
  and a missing `name="image"` attribute LinkedIn's own guidance asks for) and
  the image now serves at 2400×1260 RGB JPEG, but the final Inspector check was
  left to Scott.
- **Scott's editorial pass over the manuscript.** §10.3 credits him with
  editorial revision. He intends to do it; whether it has happened is unknown.
- **VAA prior-art citations.** §10.2 cites two Alex Chamberlain FanGraphs pieces,
  both verified by fetching the articles. Other public VAA work exists and is not
  cited.
- **The Netlify dashboard is not deployed.** `site-dashboard/` is built and
  verified locally but has not been dragged to a host.

---

## Known limitations recorded in the paper

The height adjustment removes VAA's main effect on pitch height but not the
non-linear interaction Chamberlain documented, so the §5.3 figure reads as a
conservative floor. Sequencing and tunnelling are absent entirely. Shape features
are collinear, so the block is well identified but individual coefficients are
not and should not be quoted as effect sizes. Reliability figures come from
pitchers who threw at least 250 four-seamers, which is a survivorship filter.

---

## Dashboard features added 9 September 2026

Three requests from Scott, all shipped.

**Arsenal drilldown.** Clicking any row opens the pitcher's full scored
repertoire for that season, sorted by usage, with the fastball-separation
columns showing how far each secondary sits from the pitch hitters time
against. It groups on `pid`, not name: there are two Varlands throwing in 2026
and grouping on the label would invent a repertoire neither of them has. The
panel names which pitch is the reference, and states that anything under the
150-pitch floor is not scored and therefore absent, so a pitcher with one
qualifying pitch does not read as a one-pitch pitcher.

**Shape grade, 20-80.** The shape expectation on the standard scouting scale,
50 average and 10 points per standard deviation, graded within pitch type and
season. Deliberately not a 100-scale "plus" index: that reads as Stuff+ or
PitchingBot, which are fitted on far richer inputs against different targets,
and inviting the comparison would oversell what this is. The glossary says so
outright, including that it knows nothing about location, which outranks every
shape metric here by roughly fifty to one.

**Search and CSV export.** Name filter, and a download of the whole filtered
set rather than the visible 300. Every exported row carries `edge_verdict`,
`edge_t`, `edge_pairs` and `gap_reverses_pct` for its pitch type. Once the file
is in someone else's notebook the badge and the caveat panel are gone, and an
unproven curveball edge would otherwise read exactly like a validated four-seam
one.

One thing to know if the dashboard is ever published as an Artifact rather than
hosted: the Artifact viewer sandbox blocks page-initiated downloads, so the CSV
button would silently do nothing there. On Netlify it works.

## Spin efficiency added to the model, 9 September 2026

Prompted by an outside critique arguing the work should test seam-shifted wake
(non-Magnus movement). The critique reaches the right destination by the wrong
road, and the distinction matters.

**What does not work.** Statcast's per-pitch `spin_axis` is inferred from
observed movement rather than measured off the ball -- movement direction
reproduces from it at a circular concentration of 0.91 under a clean mapping --
so a deviation built from it is close to circular by construction. What residual
survives sorts by pitch type as four-seam 12 deg, sinker 23, cutter 42, sweeper
58, slider 70, which is the gyro-spin ordering, not a seam signature. Tested
directly as a feature, a non-Magnus residual clears t=2 on **nothing**, the best
being four-seamers at t=+1.6. It is deliberately not in the model.

**What does work.** Savant's active-spin leaderboard is fetchable per pitcher,
pitch type and season (`src/spin_efficiency.py`, cached). Active spin is a
different measurement from spin rate: rate is how fast the ball turns,
efficiency is what fraction of that spin is tilted to move it. The paper's
finding that residual spin *rate* adds nothing still stands untouched.

Added to the shipped feature set, on paired CV folds over identical rows:

```
pitch       base R2   + active spin   gain        t
slider       0.1794      0.2060      +0.0265    +5.9
cutter       0.2306      0.2578      +0.0273    +3.8
four-seam    0.2187      0.2344      +0.0157    +2.6
sinker, sweeper, curveball, changeup           |t| <= 0.8
```

The three that gain are the three where gyro spin varies most, which is the
mechanism you would predict.

**It improves the expectation and barely moves the forecast.** Edge validation
went FF 12.6 -> 12.9, SI 7.5 -> 7.5, SL 5.5 -> 5.3, FC 2.4 -> 2.3. So the
"Shape exp" column is measurably more accurate, and the progression/regression
calls are not. Worth having, worth not overselling.

**Coverage constraint.** Hawk-Eye, so 2020 onward, nothing for 2015-2019, and
57% of rows overall. Survivable only because the expectation is already fit
within each season separately: a season can carry a feature its predecessors
lack without pooling across the boundary. Every row still gets a shape-only
expectation; rows with efficiency get a second, better one that overwrites it,
so nothing loses its expectation.

Revalidated after the change: backtest unchanged at 1,413 pitcher-seasons,
+0.0241 lift, 8 of 8 seasons, corr +0.239 (t=+9.2), decile spread +0.0367.
Board rows re-traced to raw parquet -- Sasaki's slider and Varland's four-seam
match exactly, and Soriano's 13-pitch discrepancy resolved to the project
requiring a non-null spin rate where the trace did not.

## Arsenal drawer and sample-size band

Added 9 September 2026.

**Nothing told anyone the rows were clickable.** The caption above the table now
opens with "Click any row to open that pitcher's full arsenal", and the drawer
itself carries a short "how to read this" block, because a movement chart is
not self-explanatory.

**The movement chart was read as a strike zone during review.** That is a fair
mistake to make of a gridded box with unlabelled axes, and it is the kind of
misreading that makes a tool untrustworthy rather than merely unclear. Fixed by
saying what it is: a "Movement, inches" title, numeric tick labels on both axes,
the origin marked "no break", direction cues on their own line, and a legend.
It also carries more now: **dashed rings mark the league average for each pitch
type in that season**, so a circle far from its ring is a pitch that behaves
unusually, and circle area tracks how often the pitch is thrown.

**The arsenal is a flyout drawer now**, not an inline card, so the table stays
in view while you read a pitcher. Scrim to click away, Escape to close, focus
returned to the row that opened it. The substance that was already there
(velocity, IVB, HB, VAA, whiff, grade, call, and separation from the fastball)
is joined by a **movement plot**: every pitch as a circle sized by usage, the
reference fastball highlighted, and a dashed line from it to each secondary
whose length is exactly the separation the model keys on. A table of separation
numbers tells you a slider is 10 mph slower and 24 inches away; the plot shows
the shape of the whole repertoire at once, which is how it is actually read.

**The parity plot has an uncertainty band** scaled to the current minimum-pitch
filter. Whiff rate is a proportion over swings, so its standard error is
sqrt(p(1-p)/swings); the band is 95% for a pitcher sitting exactly at the
filter, which makes the slider a live demonstration. Swings are estimated from
pitches using the median swing rate actually in view, not a constant, because a
sweeper draws swings at a very different rate from a sinker.

Verified against the arithmetic at three sample sizes, measured off the rendered
canvas rather than assumed:

```
filter        swings   band drawn   band predicted
250 pitches      120      8.5 pp        8.5 pp
600 pitches      287      5.4 pp        5.5 pp
1000 pitches     479      4.2 pp        4.3 pp
```

At the default 250-pitch filter the band is **+-7.2 points of whiff rate**,
which is most of the league's spread. That is the paper's reliability finding
made visible: a quarter-season sample cannot distinguish a pitcher from his own
expectation.

## A sign-convention bug worth knowing about

`src/features.py` line 90 says "positive horizontal break means arm-side for
every pitcher". **It does not.** The mirroring by handedness is correct and
both hands agree, but the resulting positive direction is *glove* side:

```
raw Statcast pfx_x   RHP sinker -1.26 ft   LHP sinker +1.28 ft
after mirroring      sinker -15.2 in       sweeper +14.0 in
```

Sinkers and changeups, which unambiguously run arm-side, come out negative;
sweepers and curveballs, which break glove-side, come out positive. That
matches Statcast's raw convention and the catcher's-view movement plots people
are used to, so the data is fine and **the model is entirely unaffected** --
the sign is consistent, so it only changes the sign of a coefficient.

What it does affect is anything that reads the name and believes it.
`pfx_x_armside`, `release_pos_x_armside` and `haa_armside` are all misnamed on
the same line of reasoning. The dashboard now labels both ends of its movement
axis and describes direction in words rather than signs, so nothing user-facing
depends on the name. Renaming the columns would touch `SHAPE_FEATURES` and mean
re-verifying every model output, so it was left alone deliberately rather than
fixed in passing.

## Searching a pitcher gives a verdict

Added 9 September 2026. The boards only ever list twenty names each way, so
searching a specific pitcher previously told you nothing about where he stood.
Now every row carries a **Call** column, and a search that lands on one pitcher
gets a plain sentence above the table: "In 2026, his most recent season here,
Louis Varland is a progression candidate on his FF."

The standing is computed from `epct`, a percentile fixed at build time within
pitch type and season, among rows clearing the 300-pitch naming line. Fixed at
build time on purpose: a verdict that changes when you drag a slider is not a
verdict. Top and bottom 10% are candidates, which is the round number nearest
the twenty-of-roughly-250 the boards actually name.

Four cases it has to get right, all handled:

- **Below the line.** No percentile at all rather than one computed against a
  pool the pitcher is not eligible for. Says so.
- **Middle of the pack.** Explicitly not a candidate, rather than silence that
  reads as a broken feature. Roughly half the population has an edge pointing
  somewhere and almost none of them are candidates.
- **Ambiguous search.** "ober" reaches eleven pitchers, only one of them Bailey
  Ober. Lists them instead of showing nothing.
- **Unvalidated pitch type.** A star on the call, because a curveball
  progression call rests on t = 1.8, not on the four-seam evidence.

The standing describes the most recent season and says so, because the arsenal
panel below it may be open on an earlier one, and two unexplained verdicts on
screen read as a bug.

## Dashboard link preview

`site-dashboard/` now carries `dashboard-card.jpg` (2400x1260 RGB JPEG) and a
full Open Graph and Twitter meta block, generated by `build_dashboard.py` from
the same payload the page embeds so the pitcher-season count on the card cannot
drift from the count in the page. The card is its own design rather than the
paper's: it leads with the actual-versus-expected scatter, because the link has
to say "this is a tool you can filter" in one glance.

`DASHBOARD_URL` in `src/config.py` is confirmed correct as of 9 September 2026.
Verified live: the card returns HTTP 200 as `image/jpeg`, 399,742 bytes, RGB
with no alpha, 2400x1260 at the 1.91:1 ratio, and the page carries
`<meta name="image" property="og:image">` in the form LinkedIn's guidance asks
for. All three faults that produced a blank card on the paper are absent here.

Unlike the paper, this URL has never been scraped before, so there is no stale
cache to bust and no `?v=2` trick needed. The first share should render.

## Where things stand

The paper is published and citable. The repo is public and in sync. The
one-pager exists as HTML and PDF with contact details. The dashboard is built,
audited, and current through 25 August 2026.

The three things most immediately outstanding:

1. ~~**The dashboard has no public home.**~~ **Done, 9 September 2026.** Live
   at https://pitch-shape-explorer.netlify.app/ with the link-preview card
   serving. Verified end to end on the deployed site: search, arsenal
   drilldown, CSV export and the grade column all work, and the search input
   is themed in both light and dark.

2. **The LinkedIn preview card is unconfirmed.** Everything upstream is fixed and
   verified serving, but LinkedIn caches aggressively and may still hold the
   broken version it scraped while the image was 404ing. Running the URL through
   Post Inspector before the next share would settle it; `?v=2` forces a fresh
   scrape if the cache proves sticky.

3. **The forecasting edge is far narrower than the paper implies, and the gap
   is now wider than it was.** After the walk-forward correction the edge
   validates on four-seamers and sinkers only, not five pitch types. The paper's
   §5.4 still reports shape as the largest block in all 14
   pitch-type-by-outcome combinations, which remains true for *explaining*
   outcomes and is unaffected by this. But a reader carries that
   fourteen-for-fourteen result into forecasting, where the honest count is two
   of seven. The paper does not currently distinguish the two, and after this
   correction that omission matters more than it did.

Beyond those, the starter-to-reliever question has come up twice and remains
untested: whether pitchers whose value concentrates in shape convert to relief
differently from those who live on command. Statcast carries inning and
times-through-order, so the data supports it. It would be a second paper.
