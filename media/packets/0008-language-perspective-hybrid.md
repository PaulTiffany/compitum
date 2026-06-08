---
packet_id: 0008
title: Language Perspective hybrid figure
asset: media/reviewed/language_perspective_hybrid_v1.svg
status: queued
required_style: light_editorial
max_text_nodes: 28
---

## 1. Intent
Create a documentation-ready hybrid figure for `docs/Language-Perspective.md`
that frames Compitum as an auditable LLM router for cs.CL reviewers, without
asking generated art to carry mathematical authority.

## 2. Source Assets
- `docs/Language-Perspective.md`
- `media/reviewed/language_perspective_scene_v1.png`
- `media/reviewed/language_perspective_hybrid_v1.svg`

## 3. Claims
- Prompts are routed across a small panel of model backends at fixed
  willingness-to-pay lambda, without a judge model.
- Policy, region, and rate constraints are enforced before selection.
- Ambiguity (gap, entropy, uncertainty) flags boundary prompts where deferral or
  conservative routing is prudent.
- Each decision emits an auditable certificate with utility components,
  constraints, and trust-region state.
- The figure is illustrative; tests and reports carry the claims.

## 4. Accessibility
Alt text: Editorial illustration of prompts routed across a small panel of models
into a certificate, with labels for prompt features, the feasible panel,
ambiguity-driven deferral, and the auditable certificate.

Long description: A text-free generated background shows prompt cards flowing
through a routing hub to a compact panel of model chips and on to a certificate
document. The deterministic overlay states the bounded claims: routing uses
lightweight prompt features, the panel is filtered by policy and region limits
before selection, ambiguous prompts are flagged for deferral, and every decision
leaves an auditable record.

## 5. Release Notes
Use near the NLP problem framing in the Language Perspective note. The bitmap
layer is illustrative; the overlay and linked tests carry the claims.
