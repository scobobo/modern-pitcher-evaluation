# Results

Four-seam fastballs, 2015–2025. 7,483,321 pitches downloaded; 2.0M four-seamers
after cleaning; ladder models fit on a pitcher-sampled 606,614-pitch subset with
5-fold `GroupKFold` by pitcher. Gains below are paired across identical folds,
so each carries a standard error.

## Headline

**The hypothesis as stated does not survive. A more precise version of it does.**

Raw spin rate, once you remove the part of it that velocity already explains,
carries essentially no independent information about run value. Velocity does.
But both are dwarfed by pitch *shape* — induced vertical break and vertical
approach angle — which is the mechanism spin was standing in for all along.

## H1 — Does spin add value beyond velocity?

### Target: run value (`delta_run_exp`, all four-seamers)

| Block added | CV R² | Incremental gain | t |
| --- | --- | --- | --- |
| controls (count, location, platoon) | 0.03667 | — | — |
| + velocity | 0.03691 | +0.00024 ± 0.00007 | **+3.63** |
| + spin (residualised) | 0.03694 | +0.00004 ± 0.00003 | +1.15 |
| + shape (IVB, HB, VAA, release) | 0.03802 | **+0.00108 ± 0.00010** | **+10.75** |

Reversing the order, so spin gets first claim on any shared variance:

| Block added | Incremental gain | t |
| --- | --- | --- |
| + spin **first** | −0.00003 ± 0.00006 | −0.46 |
| + velocity | +0.00030 ± 0.00008 | +3.63 |
| + shape | +0.00108 ± 0.00010 | +10.75 |

This is the decisive test. Spin was handed the *first* claim on all shared
variance — every advantage the design could give it — and still added nothing
distinguishable from zero. Velocity clears its noise in both orderings. Shape
adds roughly **4.5× more than velocity** and **27× more than spin**.

### Target: whiff rate on swings (1,203,525 swings)

Here the signal is much stronger, and spin does register:

| Block added | CV R² | Incremental gain | t |
| --- | --- | --- | --- |
| controls | 0.07718 | — | — |
| + velocity | 0.08298 | +0.00580 ± 0.00055 | +10.56 |
| + spin | 0.08467 | +0.00169 ± 0.00013 | +13.08 |
| + shape | 0.10419 | **+0.01952 ± 0.00076** | **+25.57** |

The two orderings give identical gains (spin +0.00169 either way), which means
that after residualisation the blocks are effectively orthogonal — the
decomposition is trustworthy.

**On whiffs, spin is real but velocity is worth 3.4× more, and shape is worth
3.4× more than velocity.** The ordering is the same on both outcomes.

### Which individual features carry the load

Permutation importance on run value, top physical features after location:
`ivb_in` (0.00097) > `release_speed` (0.00060) > `hb_in` (0.00052). In the
standardized ridge, `vaa_deg` (0.0123 runs/SD) is an order of magnitude above
`release_speed` (0.0011) and `spin_resid` (0.0012).

Note what dominates everything: `dist_from_zone_center` at 0.050 permutation
importance — 50× the best physical trait. Location is the overwhelming driver of
pitch-level run value. Any "stuff" claim needs to be read against that backdrop.

## H2 — Is velocity worth less than it used to be?

This is where your intuition has the most support, but it depends on which
question you mean.

**The physical effect (runs per mph) shows no significant trend.**
Slope −0.000125 runs/mph per year, t = −0.83 over 11 seasons. 2015 was
+0.00337, 2025 was +0.00213 — lower, but well inside the noise. Note 2018–2019
are sharp outliers toward zero, which lines up with the juiced-ball seasons.

**The whiff effect, however, has fallen hard and significantly.**
Velocity's marginal contribution to missing bats went from +0.0209 whiff
probability per mph in 2015 to −0.0022 in 2025 — slope −0.00234/year,
**t = −4.88**. Spin's whiff effect also declined (t = −2.78). Caveat worth
stating plainly: these are *partial* effects holding pitch shape constant, and
velocity is correlated with VAA and IVB. Part of this decline is shape absorbing
credit that velocity used to receive, not hitters adapting to speed.

**The league has compressed, and this is the strongest trend in the data.**

| | 2015 | 2025 | trend t |
| --- | --- | --- | --- |
| mean four-seam velocity | 93.11 mph | 94.48 mph | **+9.40** |
| SD of four-seam velocity | 2.82 mph | 2.53 mph | **−7.22** |
| velocity effect per SD (run value) | 0.0095 | 0.0054 | −1.03 |

Everyone throws hard now. The per-mph value of velocity is roughly unchanged,
but the *spread* has shrunk by 10%, so velocity separates pitchers from each
other less than it did. That is a real and defensible sense in which "velocity
matters less" — not because hitters caught up, but because the population
converged. The per-SD effect has roughly halved in point estimate (0.0094 in
2015–17 vs 0.0044 in 2023–25), though with 11 seasons that trend is not
statistically distinguishable from flat (t = −1.03).

## What this means in practice

Effect sizes are per pitch and look tiny. A starter throws roughly 1,000
four-seamers a season, so 2025's per-SD velocity effect of 0.0054 runs/pitch is
about **5.4 runs — half a win — per standard deviation of fastball velocity**.
The equivalent for residual spin is about 1.4 runs. Shape is worth several times
either.

## The recommendation

Do not pitch "spin over velocity" to a front office. The data does not support
it, and any analyst there will know that. Pitch this instead:

> Spin rate is a proxy variable. What actually predicts outcomes is the movement
> and approach angle that spin produces, and shape explains 4.5× more variance
> in run value than velocity does. Meanwhile velocity's per-mph value is stable
> but its dispersion has compressed 10% since 2015, so it is a weaker
> *differentiator* than it used to be even though it is just as valuable
> physically.

That claim is defensible, non-obvious, correctly hedged, and it is genuinely
what these 7.5M pitches say.

## Caveats

- Public Statcast cannot measure true spin efficiency; `spin_axis` is inferred
  from observed movement, not from Hawk-Eye's 3D axis. Every "spin efficiency"
  number in public analysis is an inference, including any that could be built
  here. This is the single biggest limitation.
- Observational data. Nothing here says that adding 100 rpm to a specific
  pitcher's fastball would help him.
- No pitcher random effects; command, deception, and sequencing sit in the error
  term.
- Trackman → Hawk-Eye changed spin measurement before 2020; models are fit per
  season and never pooled across that boundary, but the 2015–2019 and 2020–2025
  spin values are not perfectly comparable.
- 2020 is a 60-game season and is visibly noisy in every per-season chart.
