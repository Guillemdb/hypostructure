# Current Type II closure ledger

**Status:** no open proof rows remain for the local no-Type-II conclusion.
The controlling upgrade is `paper6:thm:unconditional-no-typeII-discharged`,
titled "No physical local Type II germ".

This tracker records the current proof interfaces in
`type_II_regularity.tex`.  Each row below is either closed directly in the
paper or routed to a closed local state-space row.  The proof uses local
compact-window estimates, canonical selected windows, and state-space
stratification.  It does not import a whole-space \(L^3\) bound, a global
profile decomposition, global Kato smallness, hidden vanishing localized
dissipation, or a noncanonical schedule.

---

## Closed Named Exits

1. **Autonomous-modulation failure / modulation-driven recentering** is closed
   by `paper6:thm:autonomous-modulation-failure-closed`, using
   `paper6a:lem:modulation-window-bound-or-exit`,
   `paper6:lem:bounded-modulation-defect-routing`,
   `paper5:thm:autonomous-modulation-selection`, and
   `paper7:def:stratified-moving-carrier-routing`.

2. **Carrier or cost-validation failure outside the canonical retained
   identity-cost channel** is closed by
   `paper6:thm:noncanonical-cost-routing-closed`, using
   `paper7:lem:no-unrecorded-cost-escape`,
   `paper7:lem:comparable-cost-compatibility`,
   `paper7:cor:retained-stratum-cost-exclusion`, and the canonical
   identity-cost fallback.

3. **Loss of retained active core / pressure-only critical mass** is closed by
   `paper6:thm:active-core-loss-closed`, using
   `paper6a:lem:terminal-local-critical-mass-routing`,
   `paper8:thm:active-profile-mass-floor`,
   `paper6a:lem:retained-frame-maximality`,
   `paper6:thm:rough-core`,
   `paper6:lem:pressure-only-critical-mass-routing`,
   `paper8:lem:harmonic-cross-pressure-routing`, and
   `paper8:cor:exterior-concentration-exit-discharged`.

The collector is `paper6:cor:remaining-named-exits-closed`; it feeds directly
into `paper6:thm:unconditional-no-typeII-discharged`.

---

## Closed Interface Rows

* Covered global-to-local entry is closed by
  `paper6:thm:physical-typeII-covered-entry` and
  `paper6:cor:covered-class-entry-discharged`.

* Compact single-core vanishing-dissipation is closed by
  `paper6:thm:single-core`, using `paper1:thm:single-core-criterion`.
  Vanishing localized dissipation must be produced on the selected sequence or
  by the retained dissipation dichotomy.

* Compact retained-core dissipation is split by
  `paper6:lem:compact-core-dissipation-dichotomy`: vanishing dissipation goes
  to the compact single-core criterion, divergent retained local cost goes to
  the compact-cost route, and first failures before retention go to closed
  named exits.

* Multibubble, multicore, same-point cascade, and gauge-degenerate retained
  branches are closed by `paper6:thm:multibubble`, via `paper3:thm:multi`,
  `paper3:cor:multi-removal`, and `paper3:prop:scale-rigidity-discharged`.

* Finite scale-collapsing cost and finite absolute scale cost are closed by
  `paper6:thm:scale-cost`, using `paper5:thm:cost-exclusion` and
  `paper5:cor:absolute-cost-exclusion`.

* Scale-rigid residual behavior is closed by
  `paper3:prop:scale-rigidity-discharged`.  Bounded selected-window limits and
  single-core dissipation must be derived locally.

* Retained compact scale-collapse is closed by
  `paper5:thm:compact-scale-collapse-rigidity-discharged` after the retained
  compact-state inputs pass.  First-failure rows from retained compact
  scale-collapse are routed through the closed named exits above.

* The canonical retained compact-cost branch is closed by
  `paper7:cor:retained-stratum-cost-exclusion`,
  `paper7:cor:ambient-adjacent-wrapper-discharged`,
  `paper6a:thm:cost-divergence-exclusion-discharged`, and
  `paper6a:cor:cost-divergence-exclusion-unconditional`.

* Terminal residual completeness on retained compact windows is closed by
  `paper6a:thm:critical-ns-profile-decomposition-discharged`.  Adjacent labels
  produced during the terminal argument are routed out of the retained terminal
  component and then closed in the assembly layer.

* Exterior concentration is closed by
  `paper8:thm:exterior-regularity-no-concentration`,
  `paper8:thm:exterior-regular-removal`,
  `paper8:thm:exterior-concentration-routed`, and
  `paper8:cor:exterior-concentration-exit-discharged`.

* Windowwise transition and positive-scale routing are proved by
  `paper7:cor:windowwise-transition-routing-holds` and
  `paper7:cor:windowwise-positive-scale-routing-holds`, using normalized
  carrier occupation states.

* Thin drift / negative scale-drift without thick windows is closed by
  `paper6a:thm:thin-negative-drift-closed`, using normalized negative-drift
  occupation blocks and canonical selected cells.

* Compactness failure on selected scale-collapse windows is routed by
  `paper5:thm:selected-compactness-failure-routed`, using
  `paper5:def:selected-scale-collapse-compactness-package`,
  `paper5:lem:selected-compactness-aubin-lions`,
  `paper5:lem:selected-compactness-pressure-only-routing`,
  `paper5:lem:selected-compactness-defect-label`, and
  `paper5:lem:selected-compactness-comparable-defect`.

---

## Reopen Criteria

Reopen this ledger only if a future edit introduces one of these defects:

* A closure uses a whole-space \(L^3\) bound, global profile decomposition, or
  global heat/Kato smallness instead of local compact-window estimates.

* A proof assumes vanishing localized dissipation instead of deriving it from
  the selected sequence, retained dissipation dichotomy, or local cost
  divergence.

* A windowwise argument replaces normalized carrier occupation states by an
  unsupported pointwise or unit-window lower bound.

* A compactness or pressure step uses weak lower semicontinuity where the paper
  requires strong local compactness or local Calderon--Zygmund pressure routing.

* A noncanonical cost diagnostic is treated as a contradiction to
  Navier--Stokes rather than being compared to the canonical identity cost or
  routed as a first failed validation row.
