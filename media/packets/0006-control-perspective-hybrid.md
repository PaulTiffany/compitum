---
packet_id: 0006
title: Control Perspective hybrid figure
asset: media/reviewed/control_perspective_hybrid_v1.svg
status: queued
required_style: light_editorial
max_text_nodes: 28
---

## 1. Intent
Create a documentation-ready hybrid figure for `docs/Control-Perspective.md`
that shows the closed-loop, judge-free control idea without asking generated art
to carry mathematical authority.

## 2. Source Assets
- `docs/Control-Perspective.md`
- `media/reviewed/control_perspective_scene_v1.png`
- `media/reviewed/control_perspective_hybrid_v1.svg`

## 3. Claims
- Boundary diagnostics (gap, entropy, uncertainty) and feasibility are measured
  at each step and emitted in a routing certificate.
- The trust-region controller shrinks the radius and caps the step size when
  drift persists.
- The metric update keeps M = L L^T + delta I positive definite by construction.
- Stability indicators are Lyapunov-inspired and operational; no formal proof is
  claimed.
- Feedback is instantaneous and judge-free; the figure is illustrative.

## 4. Accessibility
Alt text: Editorial illustration of a closed feedback loop and a contracting
trust region, with labels for measurement, trust-region control, bounded metric
update, and Lyapunov-inspired stability indicators.

Long description: A text-free generated background shows a left-to-right flow
from an instrument cluster, through feedback-loop arrows and a contracting set of
concentric circles, into a descending smooth curve. The deterministic overlay
states the bounded claims: signals are measured each step, the controller caps
steps under drift, the metric stays positive definite by construction, and the
stability indicators are operational rather than a formal proof.

## 5. Release Notes
Use near the closed-loop decomposition in the Control Perspective note. The
bitmap layer is illustrative; the overlay and linked tests carry the claims.
