# Design Notes — UI/UX Redesign (Phase 10)

## Concept: "service departure board"

A repair-shop waiting-time estimate is, conceptually, the same thing as an
airport or train departure board: a number that tells you when to expect
something. The redesign leans into that directly instead of using a generic
dashboard/card-kit look, so the visual identity is grounded in what the
product actually does.

- **Intake form** reads as a physical service ticket: a card with a
  perforated tear-line under the header, ticket-style copy ("How long's my
  repair?"), and stepper controls (instead of raw number inputs) for a more
  tactile, kiosk-like feel on mobile.
- **Result page** reads as a lit departure board: dark pine-charcoal panel,
  amber LED-style monospace numerals for the estimate, a single "flap
  reveal" animation on load (respecting `prefers-reduced-motion`), and
  board-style rows for the job details.
- **Staff dashboard** stays on light "paper" chrome for scanability at a
  desk, but keeps the same monospace numerals and status-pill language for
  visual continuity, and uses stacked cards instead of a wide table so it
  actually works on a phone.

## Tokens

- **Color:** deep pine-charcoal board (`#10201c`) with amber glyphs
  (`#f5a623`) for the result screen; pale sage paper (`#f2f4ef`) with warm
  charcoal ink (`#16201b`) elsewhere. Chosen to avoid both common AI-default
  palettes (warm-cream-plus-terracotta, and near-black-plus-neon) and to
  connect to circuit-board green / indicator-LED amber, which is genuinely
  in the subject's visual world (electronics repair).
- **Type:** IBM Plex Mono for numerals, labels, and the brand mark
  (mechanical, split-flap character); Inter for body copy and form labels.
  Two clearly distinct roles, not a decorative pairing.
- **Motion:** exactly one animated moment (the result number's flap reveal)
  rather than hover/scroll effects scattered across every element.

## Functional fix found during this pass

While testing the redesigned flow, refreshing the result page threw "Method
Not Allowed" — the old implementation rendered the result directly on the
POST response, so the browser URL stayed on `/predict`, which only accepts
POST. Fixed with a proper Post/Redirect/Get pattern: `POST /predict` now
redirects to `GET /result/<job_id>`, which looks the job up from the log and
re-renders the same page. This also means the result page is now
bookmarkable and shareable. Covered by
`tests/test_pipeline.py::test_predict_redirects_to_bookmarkable_result_url`.

## Environment note

The stylesheet loads IBM Plex Mono and Inter from Google Fonts over the
network. In an offline preview environment (no internet access) the browser
falls back to the specified system fonts (`ui-monospace` / system sans),
which still look reasonable — but the intended typefaces only render with
normal internet access, which any real deployment will have.

## What was not changed

No backend logic, model, routes' data contracts, or the ML pipeline were
touched beyond the redirect fix above — this pass is UI/UX only, per the
brief. All 7 automated tests pass after the redesign.
