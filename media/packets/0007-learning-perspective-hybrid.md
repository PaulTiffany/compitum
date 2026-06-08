---
packet_id: 0007
title: Learning Perspective hybrid figure
asset: media/reviewed/learning_perspective_hybrid_v1.svg
status: queued
required_style: light_editorial
max_text_nodes: 28
---

## 1. Intent
Create a documentation-ready hybrid figure for `docs/Learning-Perspective.md`
that frames Compitum as a constrained contextual routing problem in standard ML
terms, without asking generated art to carry mathematical authority.

## 2. Source Assets
- `docs/Learning-Perspective.md`
- `media/reviewed/learning_perspective_scene_v1.png`
- `media/reviewed/learning_perspective_hybrid_v1.svg`

## 3. Claims
- Selection maximizes a scalarized utility at fixed willingness-to-pay lambda.
- Feasibility is enforced first: filter by linear constraints and capabilities,
  then take the argmax over feasible models.
- Geometry is a low-rank SPD Mahalanobis metric updated online under a
  trust-region step cap.
- Uncertainty is calibrated with isotonic regression and quantile bounds and is
  reported as reliability, not asserted confidence.
- The figure is illustrative; tests and reports carry the claims.

## 4. Accessibility
Alt text: Editorial illustration of a learned distance geometry, a descending
loss, and a calibration curve, with labels for scalarized utility,
feasibility-first selection, learned SPD geometry, and calibrated uncertainty.

Long description: A text-free generated background shows scattered points
settling toward cluster centers inside elliptical contour rings, flowing into a
descending curve and a small rising calibration curve. The deterministic overlay
states the bounded claims: utility is scalarized at fixed lambda, feasibility is
enforced before selection, the metric is a low-rank SPD geometry updated under a
step cap, and uncertainty is calibrated and reported as reliability.

## 5. Release Notes
Use near the problem setup in the Learning Perspective note. The bitmap layer is
illustrative; the overlay and linked tests carry the claims.
