# Remaining named exits in the current Type II manuscript

**Status (latest audit): all three remaining named exits are closed.**
The paper now proves the unconditional no-Type-II theorem
`paper6:thm:unconditional-no-typeII-discharged`.

This tracker reflects the current state of `type_II_regularity.tex` after the
latest audit.

The controlling theorem in the TeX is
`paper6:thm:unconditional-no-typeII`, whose title is "Retained Type II
closure with named-exit ledger".  The label is historical; the statement says
that a physical local Type II germ either enters one of the remaining named exit
rows in `paper6:def:remaining-named-exits`, or its retained local branch is
impossible.  The three rows in `paper6:def:remaining-named-exits` are all
closed adjacent rows by the closure subsection
`paper6:subsec:remaining-exit-closures`, summarised in
`paper6:cor:remaining-named-exits-closed`, and the unconditional upgrade
is `paper6:thm:unconditional-no-typeII-discharged`, titled "No physical local
Type II germ".

The closures use local estimates and state-space stratification only.  No
closure imports a whole-space \(L^3\) bound, a global critical profile
decomposition, global Kato smallness, hidden vanishing localized dissipation,
or a noncanonical schedule.

Audit convention: some TeX labels still contain the historical prefix `ass:`
for backward reference compatibility.  They now label propositions,
definitions, or theorems, not active assumption environments.

---

## Closures of the three remaining named exits

1. **Autonomous-modulation failure / modulation-driven recentering** — closed
   by `paper6:thm:autonomous-modulation-failure-closed` via the
   modulation-window-bound-or-exit dichotomy of
   `paper6a:lem:modulation-window-bound-or-exit`, the bounded local
   modulation-defect routing lemma
   `paper6:lem:bounded-modulation-defect-routing`,
   `paper5:thm:autonomous-modulation-selection`, and the moving-carrier
   routing definition `paper7:def:stratified-moving-carrier-routing`.  The
   proof uses weak-\(*\) compactness only for normalized defect measures, not as
   a substitute for strong \(L^1\) convergence.

2. **Carrier or cost-validation failure outside the canonical retained
   identity-cost channel** — closed by
   `paper6:thm:noncanonical-cost-routing-closed` via the unrecorded-escape
   lemma `paper7:lem:no-unrecorded-cost-escape`, comparable-cost
   compatibility `paper7:lem:comparable-cost-compatibility`, retained
   stratum cost-divergence exclusion
   `paper7:cor:retained-stratum-cost-exclusion`, and canonical identity-cost
   fallback for diagnostics that are not explicitly comparable.  Incomparable
   noncanonical diagnostics are not treated as contradictions to the
   Navier--Stokes class.

3. **Loss of retained active core / pressure-only critical mass** — closed
   by `paper6:thm:active-core-loss-closed` via the terminal critical-mass
   routing lemma `paper6a:lem:terminal-local-critical-mass-routing`, the
   active mass-floor `paper8:thm:active-profile-mass-floor`, retained-frame
   maximality `paper6a:lem:retained-frame-maximality`, the rough-core
   reduction `paper6:thm:rough-core`, the pressure-only local routing lemma
   `paper6:lem:pressure-only-critical-mass-routing`, the harmonic
   cross-pressure routing `paper8:lem:harmonic-cross-pressure-routing`, and
   the exterior concentration discharge
   `paper8:cor:exterior-concentration-exit-discharged`.

---

## Historical: previously closed rows

The following rows are closed in the present manuscript, subject to the audit
boundaries listed here.

* Covered entry/global-to-local entry is discharged by
  `paper6:thm:physical-typeII-covered-entry` and
  `paper6:cor:covered-class-entry-discharged`.  This proves that a physical
  local Type II germ enters the local state-space taxonomy or reaches a named
  exit; it does not eliminate the named exits.

* Compact single-core vanishing-dissipation is closed by
  `paper6:thm:single-core`, using `paper1:thm:single-core-criterion`.
  The audit boundary is that vanishing localized dissipation must be produced
  on the original selected sequence or by the retained dissipation dichotomy,
  not assumed.

* Compact retained-core dissipation is split by
  `paper6:lem:compact-core-dissipation-dichotomy`.  Vanishing dissipation goes
  to the compact single-core criterion; divergent retained local cost goes to
  the compact-cost/cost-divergence route; first failures before retention are
  named exits.

* Multibubble, multicore, same-point cascade, and gauge-degenerate retained
  branches are closed by `paper6:thm:multibubble`, via `paper3:thm:multi`,
  `paper3:cor:multi-removal`, and the scale-rigidity discharge.

* Finite scale-collapsing cost and finite absolute scale cost are closed by
  `paper6:thm:scale-cost`, using `paper5:thm:cost-exclusion` and
  `paper5:cor:absolute-cost-exclusion`.

* Scale-rigid residual behavior is closed by
  `paper3:prop:scale-rigidity-discharged`.  The audit boundary is unchanged:
  bounded selected-window limits and single-core dissipation must be derived
  locally, not imported.

* Retained compact scale-collapse is closed by
  `paper5:thm:compact-scale-collapse-rigidity-discharged`, after the retained
  compact-state inputs have passed.  Failures of those inputs are not closed by
  this theorem; they are the scale-collapse named exits listed below.

* The canonical retained compact-cost branch is closed by
  `paper7:cor:retained-stratum-cost-exclusion`,
  `paper7:cor:ambient-adjacent-wrapper-discharged`,
  `paper6a:thm:cost-divergence-exclusion-discharged`, and
  `paper6a:cor:cost-divergence-exclusion-unconditional`, on the canonical
  windowwise schedule and retained identity-cost channel.

* Terminal residual completeness on retained compact windows is discharged by
  `paper6a:thm:critical-ns-profile-decomposition-discharged`.  This is local
  terminal state-space completeness; adjacent labels produced during the
  terminal argument are routed out of the retained terminal component, not
  globally eliminated inside that theorem.

* Exterior concentration and noncompact exterior concentration are no longer
  independent remaining exits.  No-exterior-concentration annuli are regular
  and removable by `paper8:thm:exterior-regularity-no-concentration` and
  `paper8:thm:exterior-regular-removal`; the concentration side is routed by
  `paper8:thm:exterior-concentration-routed` and
  `paper8:cor:exterior-concentration-exit-discharged`.  The closure is a local
  routing closure: exterior profiles decouple from compact core frames,
  separated/multicore/cascade subtypes go to the multibubble package,
  exterior scale-collapse is rebased into the ordinary scale-collapse ledger,
  noncompact exterior branches are reduced to dyadic active carriers or regular
  dyadic no-concentration tails.  Fixed-annulus stress is first passed through
  `paper8:lem:fixed-annular-stress-ckn-dichotomy`, so regular annular stress is
  not silently promoted to a singular carrier.  Pressure-only exterior
  detection is routed by the far-field harmonic-tail/dyadic-carrier lemma or by
  the ordered non-exterior pressure row.  Reopen only if the proof uses a
  whole-space \(L^3\) bound, a global profile decomposition, global Kato
  smallness, hidden vanishing localized dissipation, or a cardinality bound for
  exterior carriers not supplied by local finite-overlap/Vitali selection.

* Windowwise transition and positive-scale routing are no longer primitive
  assumptions.  They are proved by
  `paper7:cor:windowwise-transition-routing-holds` and
  `paper7:cor:windowwise-positive-scale-routing-holds` using normalized carrier
  occupation states.  This proves routing of \(J_1\) and \(J_7\), not automatic
  elimination of every earlier exterior or modulation label; negative-drift
  labels are handled separately by the thin-drift closure and the retained
  scale-collapse routing.

* Thin drift / negative scale-drift without thick windows is no longer an
  independent remaining exit.  It is closed by
  `paper6a:thm:thin-negative-drift-closed`, using normalized negative-drift
  occupation blocks.  Persistent fixed-length drift windows return to the
  thick-window ledger by finite overlap with the canonical validated unit-window
  schedule; genuinely diffuse blocks contain canonical selected cells on which
  the negative scale term is a vanishing local \(H^{-1}\) perturbation and route
  the branch to another named row or to a closed retained row.  Reopen only if
  the proof validates arbitrary terminal intervals instead of canonical schedule
  windows, silently converts diffuse drift into a pointwise/unit-window lower
  bound, or uses a global \(L^3\) or whole-space profile estimate.

* Compactness failure on selected scale-collapse windows is no longer an
  independent remaining exit.  It is routed by
  `paper5:thm:selected-compactness-failure-routed`, using the local selected
  compactness package
  `paper5:def:selected-scale-collapse-compactness-package`, local
  Aubin--Lions compactness
  `paper5:lem:selected-compactness-aubin-lions`, pressure-only routing
  `paper5:lem:selected-compactness-pressure-only-routing`, first-failure
  defect labelling `paper5:lem:selected-compactness-defect-label`, and
  comparable-defect grouping
  `paper5:lem:selected-compactness-comparable-defect`.  The closure is a
  routing closure: a compactness-package failure is either repaired into local
  strong compactness on the canonical selected windows or assigned to a more
  specific row such as exterior/separated concentration, autonomous modulation,
  retained active-core loss, pressure-only critical mass, rough-core loss, or
  multicore/cascade.  Reopen only if the proof uses arbitrary noncanonical
  windows, a global \(L^3(\mathbb R^3)\) bound, a whole-space profile
  decomposition, hidden vanishing localized dissipation, or pressure compactness
  by weak lower semicontinuity instead of the local pressure/exterior routing.

---

## Discharged exits — audited proof boundaries

The three rows that the previous audit listed as remaining are now closed
adjacent rows in `paper6:subsec:remaining-exit-closures`.

* Autonomous-modulation failure / modulation-driven recentering is closed by
  `paper6:thm:autonomous-modulation-failure-closed`.  The bounded-modulation
  case is hardened by `paper6:lem:bounded-modulation-defect-routing`: the proof
  does not use Dunford--Pettis as a false strong \(L^1\) compactness theorem.
  Either a constant \(L^1\) modulation limit exists, contradicting the
  autonomous-modulation alternative, or a normalized local modulation-defect
  occupation state is recorded as modulation-driven recentering, positive
  scale-shell work, negative scale-drift, active-core loss, or repaired-gauge
  degeneracy.

* Carrier/cost validation failure outside the canonical retained channel is
  closed by `paper6:thm:noncanonical-cost-routing-closed`.  The proof does not
  declare an incomparable noncanonical diagnostic to be a contradiction to the
  Navier--Stokes class.  Instead, incomparable diagnostics are not used to
  retain the final branch: the ordered proof falls back to the canonical
  identity-cost channel or records the first failed local validation row.
  Carrier-subidentity and finite-error failures are routed through the
  componentwise local carrier diagnostics, not by inserting a global tail
  estimate.

* Loss of retained active core / pressure-only critical mass is closed by
  `paper6:thm:active-core-loss-closed`.  The pressure-only subcase is hardened
  by `paper6:lem:pressure-only-critical-mass-routing`: the proof does not infer
  \(\|V_n\|_{L^3}\to0\) from a sub-active mass floor.  The local pressure atlas
  first tests the localized Calderon--Zygmund source.  If that source has
  positive compact-window stress, terminal critical-mass routing records a
  velocity-supported adjacent label; if the source vanishes, the remaining
  harmonic pressure is routed by `paper8:lem:harmonic-cross-pressure-routing`.

The corollary `paper6:cor:remaining-named-exits-closed` collects these three
closures, and `paper6:thm:unconditional-no-typeII-discharged` upgrades the
controlling retained-closure theorem to the stated no-physical-local-Type-II
conclusion.

---

## Retired audit items

The following former assumptions are not active standalone assumptions, but they
should be reopened if their audit boundaries fail.

* `paper3:ass:scale-rigidity` is retired by
  `paper3:prop:scale-rigidity-discharged`.  Reopen only if bounded
  selected-window limits or single-core dissipation are being assumed rather
  than proved locally.

* `paper7:def:windowwise-transition-routing` and
  `paper7:def:windowwise-positive-scale-routing` are retired as primitive
  assumptions.  Reopen only if the proof stops using normalized carrier
  occupation states and silently replaces nonintegrability by a pointwise or
  unit-window lower bound.

* `paper6a:ass:cost-divergence-exclusion` is retired on the canonical
  windowwise schedule.  Reopen only if the proof uses cost divergence before
  the carrier, windowwise routing, and named-exit ordering rows have been
  verified.

* `paper6a:ass:critical-ns-profile-decomposition` is retired as a global
  profile-decomposition assumption.  The replacement is local terminal
  state-space completeness on compact windows.  Reopen only if global profile
  decomposition, global heat/Kato smallness, or hidden whole-space boundedness
  reappears.

* Compact scale-collapse stationary rigidity is retired on retained compact
  scale-collapse branches.  The proof uses
  `paper5:thm:compact-scale-collapse-rigidity-discharged`, not an unstated
  stationary omega-limit or self-similar Liouville theorem.  The first-failure
  exits from retained compact scale-collapse are handled by the closed-exit
  rows recorded above; reopen only if a future edit uses one of those
  first-failure rows without routing it through its local closure theorem.

---

## Suggested closure order

All three rows are closed.  No further action is required for the local
no-Type-II conclusion.  Reopen this list only if one of the audit boundaries
recorded above is violated by a future edit.
