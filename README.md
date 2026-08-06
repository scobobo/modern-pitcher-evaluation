# The Shape of the Modern Pitch

Analysis code for a pitch-level study of **7,483,321 Statcast pitches (2015–2025)**
testing what actually drives pitch outcomes — and what that implies for how
pitchers should be evaluated.

📄 **Read the paper:** https://the-shape-of-the-modern-pitch.netlify.app/

💻 **Code:** [https://github.com/scobobo/modern-pitcher-evaluation](https://github.com/scobobo/modern-pitcher-evaluation)

---

## Findings

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

## What this study does not claim

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

## Layout

```
src/config.py         season windows, pitch groups, constants
src/fetch.py          Statcast pulls, cached one parquet per season
src/features.py       approach angles, movement, spin residualisation, outcomes
src/model.py          nested feature-block ladder, grouped CV, permutation importance
src/temporal.py       per-season effects and trend tests
src/evaluation.py     reliability, year-over-year stability, next-season forecasting
src/plots.py          analysis figures
src/paper_figures.py  publication figures

run_analysis.py         attribution ladder (§5.1)
run_paper_analysis.py   VAA robustness, pitch-type generality, reliability (§5.3–5.5)
run_sample_size_test.py the crossover experiment (§5.6)
build_docx.py           Word edition
build_standalone.py     standalone HTML edition
```

## Reproducing

```bash
pip install pandas numpy scikit-learn matplotlib pyarrow
pip install --no-deps pybaseball pygithub pyjwt Deprecated wrapt pynacl
pip install requests beautifulsoup4 lxml tqdm attrs python-dateutil cffi
```

`pybaseball` pulls in `cryptography` through a Retrosheet module this project
never uses; the two-step install avoids that build.

```bash
# ~7.5M pitches, one parquet per season (not committed — 653 MB)
python src/fetch.py --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025

python run_analysis.py                  # §5.1 attribution ladder
python run_analysis.py --target is_whiff
python run_paper_analysis.py            # §5.3–5.5
python run_sample_size_test.py          # §5.6 crossover
```

Results write to `output/` as CSV.

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

The reporting function prints "hypothesis not supported" when the numbers say so.
It was written before the results were known and was not adjusted afterward.

## License

MIT — see [LICENSE](LICENSE).
