---
packet_id: 0009
title: Statistical Notes hybrid figure
asset: media/reviewed/statistical_notes_hybrid_v1.svg
status: queued
required_style: light_editorial
max_text_nodes: 28
---

## 1. Intent
Create a documentation-ready hybrid figure for `docs/Statistical-Notes.md` that
shows the evaluation methodology for stat.ML reviewers, without asking generated
art to carry mathematical authority.

## 2. Source Assets
- `docs/Statistical-Notes.md`
- `media/reviewed/statistical_notes_scene_v1.png`
- `media/reviewed/statistical_notes_hybrid_v1.svg`

## 3. Claims
- Regret is paired per evaluation unit before aggregation to reduce variance.
- Confidence intervals use a paired nonparametric bootstrap with 1000 resamples
  and 95% percentile intervals.
- Calibration is reported via reliability curves and a Spearman rank correlation
  between uncertainty and absolute regret.
- The coherence prior is a whitened KDE with clipping, and dispersion uses
  Ledoit-Wolf shrinkage.
- The figure is illustrative; tests and reports carry the claims.

## 4. Accessibility
Alt text: Editorial illustration of a bootstrap distribution, an interval plot,
and a reliability curve, with labels for paired regret, bootstrap intervals,
calibration, and bounded priors.

Long description: A text-free generated background shows a smooth bell-shaped
distribution, a dot-and-whisker interval, and a rising reliability curve along a
faint diagonal. The deterministic overlay states the bounded claims: regret is
paired before aggregation, intervals come from a paired bootstrap, calibration is
reported via reliability curves and rank correlation, and the priors are bounded
by whitening, clipping, and shrinkage.

## 5. Release Notes
Use near the methodology summary in the Statistical Notes note. The bitmap layer
is illustrative; the overlay and linked tests carry the claims.
