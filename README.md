# The Shape of the Modern Pitch

Analysis code for a pitch-level study of **7,483,321 Statcast pitches (2015–2025)**
testing what actually drives pitch outcomes — and what that implies for how
pitchers should be evaluated — and for **Pitch Shape Explorer**, a dashboard
that applies the finding to every MLB pitcher-season from 2015 through 2026.

📄 **Read the paper:** https://the-shape-of-the-modern-pitch.netlify.app/

📊 **Explore the dashboard:** https://pitch-shape-explorer.netlify.app/

💻 **Code:** [https://github.com/scobobo/modern-pitcher-evaluation](https://github.com/scobobo/modern-pitcher-evaluation)

---

## The paper

### Findings

- **Pitch shape** — induced vertical break, horizontal break, and vertical
  approach angle — adds **4.5× more** cross-validated explanatory power than
  velocity for run value, and **3.4× more** for whiffs.
- **Residual spin rate adds nothing** distinguishable from zero once velocity is
  partialled out — even when handed *first* claim on the shared variance
  (−0.00003 ± 0.00006, t = −0.46).
- Shape is the largest block in **all 14** pitch-type-by-outcome combinations.
- **Shape metrics are measured almost exactly** (split-half r ≈ 0.99) while run
  value per pitch is barely measured at all (r = 0.20). That asymmetry, not the
  effect sizes, is what changes evaluation.
- Below **~80 pitches**, a pitcher's shape predicts his next season better than
  his own results do. Above that, results win — but shape keeps adding.

### What this study does not claim

Stated up front because they constrain the thesis:

- **Command outranks shape by roughly 50×.** Distance from the middle of the zone
  scores 0.050 in permutation importance; the best physical trait scores 0.00097.
  Shape dominates among *pitch-intrinsic* properties — not among all determinants
  of a pitch's outcome.
- **Shape does not replace outcome data on full seasons.** At 500 pitches, past
  results predict next-season whiff rate at R² = 0.362 versus shape's 0.188.
- The study **began from the opposite hypothesis** — that spin had displaced
  velocity — and rejected it. See §6 of the paper, which also documents an
  experimental design error that had to be corrected.

## The dashboard

**Pitch Shape Explorer** scores **15,358 pitcher-seasons** — one row per pitcher,
season and pitch type with at least 150 pitches, 2015 through the end of the 2026
regular season — against one question: given how this pitch moves, how often
should hitters miss it when they swing?

- Filter by season, age, pitch type, handedness, and minimum pitches.
- Search any pitcher for a one-line verdict on his most recent season.
- Click a row for his full arsenal: each pitch he threw 150+ times, plotted by
  movement against the league average for that pitch type, and how far each one
  sits from his fastball.
- Each pitch carries a **shape grade** on the 20–80 scouting scale, graded within
  pitch type and season.
- Export the filtered rows to CSV. Every row carries its pitch type's validation
  verdict, so an unproven call cannot pass for a proven one once the file leaves
  the page.

### How it works

Two models, both fit once and applied to every row. Filtering changes which
pitchers you see, never what either model learned.

- **Shape expectation.** The whiff rate a pitch's movement, approach angles,
  velocity, release and extension predict. Secondary pitches are also measured
  against the pitcher's primary fastball (velocity, movement and release
  separation), since a slider misses bats by differing from the fastball hitters
  are timing. From 2020 it adds Savant's spin efficiency. Fit within each season,
  out of fold and grouped by pitcher, so no pitcher's expectation comes from a
  model that has seen him.
- **Edge.** How much the shape expectation adds to a forecast of next season's
  whiff rate beyond what this season's whiff rate alone predicts, i.e. beyond
  regression to the mean. Trained on the four most recent complete seasons
  (2023–2026) rather than every season back to 2015, because the game drifts and
  older seasons made recent forecasts worse. The top and bottom 10% by edge, among
  rows with 300+ pitches, are the progression and regression candidates.

### Validation

Tested walk-forward: each season is scored using only seasons that had already
closed, against the next complete season.

On four-seamers (`run_leaderboard.py`), shape beat regression to the mean in
**9 of 9** seasons tested, across 1,617 out-of-sample pitcher-seasons. The
correlation between the predicted edge and what regression to the mean missed is
+0.235 (t = +9.7). The top 10% by edge beat their mean-reversion forecast by
**2.0 points** of whiff rate the next season; the bottom 10% fell short by 2.0.

By pitch type, with nothing called validated below t = 2:

| Pitch | Verdict | t | Season pairs |
| --- | --- | --- | --- |
| Four-seam | validated | 12.9 | 2,299 |
| Sinker | validated | 7.2 | 1,205 |
| Slider | validated | 5.5 | 969 |
| Cutter | validated | 2.4 | 128 |
| Changeup | validated | 2.1 | 415 |
| Curveball | promising | 1.7 | 252 |
| Sweeper | untested | — | 0 |

Changeups and cutters only just clear the bar. Only complete seasons are scored,
so verdicts change when a season closes, not with each data refresh.

### What the dashboard does not claim

- **Not every gap closes.** About 45% of a gap between results and shape closes
  the following season (36–49% depending on the pitch). The rest usually reflects
  command or deception, which the tracking cannot see.
- **It knows nothing about location**, which outranks every shape trait by
  roughly 50× per the paper. Shape sets the ceiling; command decides how much of
  it a pitcher reaches.
- **Seven pitch types are modelled**: four-seam, sinker, cutter, slider, sweeper,
  curveball, and changeup. Knuckle-curves and splitters are not.

## Layout

```
src/config.py           season windows, pitch groups, constants
src/fetch.py            Statcast pulls, cached one parquet per season
src/features.py         approach angles, movement, spin residualisation, outcomes
src/model.py            nested feature-block ladder, grouped CV, permutation importance
src/temporal.py         per-season effects and trend tests
src/evaluation.py       reliability, year-over-year stability, next-season forecasting
src/plots.py            analysis figures
src/paper_figures.py    publication figures and link-preview cards

src/projection.py       shape expectation, fastball separation, gap reversal
src/leaderboard.py      walk-forward backtest, decile analysis, board builder
src/spin_efficiency.py  Savant active spin, cached
src/players.py          birth dates and handedness from MLB StatsAPI, cached

run_analysis.py           attribution ladder (§5.1)
run_paper_analysis.py     VAA robustness, pitch-type generality, reliability (§5.3–5.5)
run_sample_size_test.py   the crossover experiment (§5.6)
build_docx.py             Word edition
build_standalone.py       standalone HTML edition

build_dashboard_data.py   dashboard dataset and per-pitch-type verdicts
build_dashboard.py        injects it into dashboard_template.html -> site-dashboard/
run_leaderboard.py        four-seam walk-forward backtest
run_projection.py         regression and progression candidates
```

## Reproducing

Tested on CPython 3.12.

```bash
pip install pandas numpy scikit-learn matplotlib pyarrow
pip install --no-deps pybaseball pygithub pyjwt Deprecated wrapt pynacl
pip install requests beautifulsoup4 lxml tqdm attrs python-dateutil cffi python-docx pypdf
```

`pybaseball` pulls in `cryptography` through a Retrosheet module this project
never uses; the two-step install avoids that build.

**The paper:**

```bash
# ~7.5M pitches, one parquet per season (not committed — 653 MB)
python src/fetch.py --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025

python run_analysis.py                  # §5.1 attribution ladder
python run_analysis.py --target is_whiff
python run_paper_analysis.py            # §5.3–5.5
python run_sample_size_test.py          # §5.6 crossover
```

Results write to `output/` as CSV.

**The dashboard:**

```bash
python build_dashboard_data.py   # prints the per-pitch-type verdicts
python build_dashboard.py        # writes site-dashboard/ (page + link-preview card)
python run_leaderboard.py        # four-seam walk-forward backtest
```

The first run downloads anything not already cached: 2026 Statcast, Savant's
active-spin leaderboard and pitcher birth dates. `site-dashboard/` is
self-contained; upload the whole folder to any static host.

## How the analysis guards against fooling itself

| Trap | Guard |
| --- | --- |
| Spin/velocity collinearity | Spin residualised on velocity; ladder run in both orderings |
| Pitcher memorisation | `GroupKFold` on pitcher — no pitcher in both train and test |
| Differing missingness between models | Every rung scored on one common complete-case sample |
| Noise read as signal | Gains paired across identical folds; nothing called real below t = 2 |
| Shape as laundered location | VAA residualised on plate height; 93% of the effect survives |
| Spin meaning different things per pitch | Residualisation and modelling within pitch type |
| Trackman → Hawk-Eye change (2020) | Per-season models, never pooled across the boundary |
| Dashboard forecasts judged on their own future | Walk-forward: each board scored by a model trained only on earlier, complete seasons |
| Verdicts moving with partial-season noise | Only complete seasons are scored; verdicts change when a season closes |
| Extremes made of thin samples | Named candidates need 300+ pitches; expectations clipped to the observed range |
| A filter quietly refitting the model | Both dashboard models fit once and applied to every row |

The reporting function prints "hypothesis not supported" when the numbers say so.
It was written before the results were known and was not adjusted afterward.

## License and citation

Code: MIT — see [LICENSE](LICENSE).

Paper text and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) © 2026 Scott Luntz.

**Cite as:** Luntz, S. (2026). *The Shape of the Modern Pitch.* Zenodo. https://doi.org/10.5281/zenodo.22037431

Archived code release: [10.5281/zenodo.22037409](https://doi.org/10.5281/zenodo.22037409)
