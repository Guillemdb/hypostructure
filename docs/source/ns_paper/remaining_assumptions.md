# Remaining assumptions in the current Type II manuscript

This tracker records the assumptions and named proof obligations in the current
`type_II_regularity.tex` manuscript, including retired audit items and any items
that would be reopened if an audit fails.  The controlling goal
for this file is an unconditional no-Type-II theorem, not merely exclusion on a
preselected local branch class.

The current manuscript now proves the no-Type-II statement in the local
state-space framework.  The former outer quantifier gap has been closed by
three branchwise wrappers: physical covered-class entry, ambient
adjacent-stratum/carrier applicability, and expanded terminal hypotheses.  Every
possible physical local Type II singular branch is shown to enter the covered
local path, or else its first failure is routed to an explicitly named local
alternative whose closure is supplied by the local packages below.  The audit
boundary remains important: none of these wrappers may import a hidden global
\(L^3\) estimate, global profile decomposition, global Kato smallness, or
unstated vanishing localized dissipation.

The cost-exclusion pass changed the status of the Section 7 cost layer:

* `paper7:def:ns-cost-to-exclusion-data` is now a data condition, not a
  primitive assumption.
* `paper7:def:cost-exclusion-data` is still a labelled interface in the TeX
  file, but it is verified on the retained compact stratum by
  `paper7:prop:local-cost-exclusion-data-verification` and therefore is not
  counted here as a remaining assumption.
* `paper7:cor:retained-stratum-cost-exclusion` gives the actual retained
  compact cost-divergence exclusion.
* `paper7:def:classwise-retained-compact-coverage` is now discharged by
  `paper7:thm:classwise-retained-compact-coverage`, using the ordered validation
  theorem `paper7:lem:no-unrecorded-cost-escape`.
* `paper7:def:compact-cost-adjacent-closure` has been reduced further by
  `paper7:thm:adjacent-closure-reduced-to-carrier-completion`: after
  non-carrier exits are routed to their named structural/interface strata, the
  Section 7-specific residual is carrier-estimate completion, and in the
  canonical or energy-controlled cost channel this is reduced through
  finite-error carrier selection to `paper7:def:annular-finite-error-selection`.
  The new component checklist
  `paper7:def:componentwise-annular-finite-error` isolates the six fixed and
  seven moving local error estimates seen by the carrier test.  The current
  sharp reduction now runs through the canonical windowwise moving family:
  `paper7:def:moving-test-family`,
  `paper7:lem:successful-moving-test-family`,
  `paper7:lem:windowwise-tight-radius-schedule`,
  `paper7:lem:windowwise-shell-pressure-translation`,
  `paper7:cor:windowwise-only-transition-and-scale-remain`,
  `paper7:def:windowwise-transition-routing`,
  `paper7:def:windowwise-positive-scale-routing`,
  `paper7:def:normalized-carrier-blocks`,
  `paper7:lem:closed-carrier-diagnostics`,
  `paper7:lem:carrier-state-compactness-labels`,
  `paper7:lem:nonintegrable-component-certificate`,
  `paper7:lem:transition-carrier-label-is-exterior`,
  `paper7:lem:transition-carrier-state-stratification`,
  `paper7:lem:positive-scale-carrier-core-realization`,
  `paper7:thm:positive-scale-certificate-enters-scale-rigid`,
  `paper7:lem:positive-scale-work-stratification`,
  `paper7:cor:windowwise-transition-routing-holds`,
  `paper7:cor:windowwise-positive-scale-routing-holds`,
  `paper7:thm:windowwise-routing-gives-stratified-routing`,
  `paper7:thm:stratified-moving-carrier-routing`, and the direct closure
  corollary `paper7:cor:adjacent-closure-reduced-to-windowwise-routing`.
  The two windowwise routing definitions are now discharged in the TeX by
  normalized carrier-state extraction and local state-space labelling, so they
  are not counted as active assumptions.  The older four-statement theorem
  `paper7:thm:small-routing-statements-give-stratified-routing` remains in the
  file as a more general fallback, but it is no longer the sharp residual.  In
  the same pass, the scale-cutoff drift term was reduced from a full
  \(|a|\)-weighted shell error to its genuinely bad \(a_+\)-part for radial
  nonincreasing cutoffs, and the pressure-flux item was reduced by the
  annular estimate `paper7:lem:annular-pressure-flux` to a local enlarged-shell
  \(L^3\) bound plus an \(R^{-1}\) harmonic-tail term.  The canonical
  reformulation is now: on the windowwise moving family, viscous cutoff,
  convection, pressure, and translation are automatic; nonintegrable failure
  of the transition term is read as a transition carrier state labelled by
  exterior/leakage; nonintegrable failure of the positive scale-shell term is
  read as positive scale-work carrier state routed to scale-rigidity unless an
  earlier exterior, modulation-driven recentering, or negative-drift label has
  already fired; and the negative scale-drift term remains routed to Section 6.
* `paper8:ass:exterior-regularity` has been replaced by the local annular
  exterior concentration alternative: no-exterior-concentration annuli are
  regular by `paper8:thm:exterior-regularity-no-concentration`, while failed
  annular tests enter the exterior concentration alternative
  `paper8:cor:exterior-annulus-alternative`.
* `paper8:ass:terminal-windowed-profile-completeness` is also not listed
  separately, because it is now a theorem derived from the same local terminal
  state-space completeness package that discharges
  `paper6a:ass:critical-ns-profile-decomposition`.
* `paper6a:ass:cost-divergence-exclusion` is now discharged by
  `paper6a:thm:cost-divergence-exclusion-discharged` and
  `paper6a:cor:cost-divergence-exclusion-unconditional`; it is retained below
  only as an audit item, not as an active remaining assumption.

When an item is active, it is read in the same way:

1. what it literally says,
2. what it means conceptually,
3. what role it plays in the argument,
4. what kind of theorem would be needed to discharge it,
5. how strong it is,
6. how it relates to the other assumptions.

Retired items may still be recorded below as audit notes, but they are not
counted as open assumptions unless the audit fails and the item is reopened.

---

# Current status for the unconditional no-Type-II theorem

No active proof obligation remains for the no-Type-II theorem in the local
state-space framework.  The TeX now contains the final branchwise assembly
theorem `paper6:thm:unconditional-no-typeII`, and
`paper6:thm:local-typeII-exclusion` cites it to state that no singular point
lies in the local Type II alternative.  The Type I side is separate and is not
needed for the no-Type-II claim.

## Discharged: covered-class entry / global-to-local coverage

Current status: discharged by the physical-entry package in
`type_II_regularity.tex`:

* `paper6:def:physical-local-typeII-germ` defines the physical local Type II
  germ directly at a singular point: \(x_0\in\Sigma(T)\), the local Type I
  bound fails, and the germ is interior to the Navier--Stokes domain.
* `paper6:lem:physical-typeII-germ-admissible` turns such a germ, after harmless
  translation and terminal restriction, into an admissible branch in
  `def:c1-admissible-typeII-class`.
* `paper6:lem:physical-typeII-germ-concentration-sequence` uses the pointwise
  CKN concentration dichotomy to produce a positive local Type II concentration
  sequence.
* `paper6:lem:no-extra-outside-class-row` applies the ordered C1 discharge to
  show that any analytic-data failure is class-incompatible or a named
  non-retained local state-space exit.
* `paper6:thm:physical-typeII-covered-entry` proves that every physical local
  Type II germ enters `paper6:thm:state-decomposition`, unless it exits earlier
  through one of those named local alternatives.
* `paper6:cor:covered-class-entry-discharged` records the discharge.
* `paper6:rem:covered-entry-scope` records the scope: this is an entry theorem,
  not by itself the final unconditional no-Type-II theorem, and it does not
  supply the ambient adjacent-closure or expanded terminal wrappers.  It also does not
  claim that every named non-retained exit has already been eliminated; it
  records that such exits are in the taxonomy and must be closed by the retired
  local packages or by the now-proved ambient/terminal wrappers.

The proof is local.  It uses pointwise CKN concentration, the local Type
I/Type II split, ordered first-failure routing, and selected-window
state-space decomposition.  It must not silently use:

* global \(L^3(\mathbb R^3)\) boundedness;
* global profile decomposition;
* global Kato smallness;
* global compactness of the original solution;
* vanishing localized dissipation not produced by the branch selection itself.

Audit result after the latest review: the discharge is non-circular.  It does
not invoke `paper6:thm:local-typeII-exclusion` or the final state-elimination
proposition.  It invokes `paper6:thm:state-decomposition` only to place a
retained positive local Type II concentration sequence into the local taxonomy.
The closure of the resulting alternatives remains governed by the retired local
packages and the two discharged wrappers below.

## Discharged: ambient adjacent-stratum closure and carrier applicability

Current status: discharged by the branchwise ambient wrapper in
`type_II_regularity.tex`:

* `paper7:lem:ambient-noncarrier-exits-close` proves that every non-carrier
  first failure listed by `paper7:thm:noncarrier-exits-route` is either
  class-incompatible or assigned to a named local state-space alternative
  already closed by the local packages.
* `paper7:lem:ambient-carrier-applicability` verifies that a retained carrier
  branch uses the canonical identity-cost channel, where energy-control is the
  identity comparison; a non-identity cost without the needed upper comparison
  is routed to the ordered cost-compatibility/carrier-comparison exit.
* `paper7:lem:ambient-negative-drift-branch` identifies \(J_6\)-failure with
  the negative scale-drift branch only after the local retained scale-collapse
  prerequisites have passed; failure of any prerequisite is first routed to a
  named local exit.  The retained case is then handled by the local scale-drift,
  scale-collapse, and compact scale-collapse packages, without feeding adjacent
  closure back into itself.
* `paper7:thm:ambient-adjacent-closure-branchwise` proves that the hypotheses of
  `paper7:cor:adjacent-closure-reduced-to-windowwise-routing` hold for every
  physical local Type II branch reaching the compact-cost validation layer.
* `paper7:cor:ambient-adjacent-wrapper-discharged` records the discharge.
* Latest audit hardening: `paper7:lem:retained-pressure-tail-windowwise` now
  handles the \(J_4\) pressure flux in the canonical windowwise family by
  local pressure/exterior routing and nested-ball harmonic tails, rather than
  by importing a whole-space \(L^3\) bound.

The two windowwise routing definitions remain retired:
`paper7:cor:windowwise-transition-routing-holds` and
`paper7:cor:windowwise-positive-scale-routing-holds` discharge them on the
canonical moving family.  The new ambient wrapper supplies the missing outer
quantifier and carrier applicability.

## Discharged: expanded terminal hypotheses as branchwise theorems

Current status: discharged by:

* `paper6a:thm:expanded-terminal-hypotheses-branchwise` — every physical local
  Type II branch after covered entry either exits through a named local
  alternative already eliminated by the local state-space packages, or the nine
  items of `paper6a:def:expanded-terminal-data` hold on a retained terminal
  tail.
* `paper6a:lem:terminal-local-critical-mass-routing` — the finite critical-mass
  input used in the terminal wrapper is local on compact terminal windows:
  failure of the lower active mass floor is non-retention, and failure of a
  finite local \(L^3\) upper bound produces a comparable retained label or a
  named adjacent active label.
* `paper6a:cor:expanded-terminal-wrapper-discharged` — records that the
  expanded terminal hypotheses are no longer active assumptions for the
  no-Type-II theorem.

The wrapper covers local alternative decomposition, critical tightness,
terminal compactness and finite critical mass, perturbative residual stability,
terminal state-space completeness, exterior-component removal, repaired-gauge
representation, local Caccioppoli/windowed regularity, and compact
scale-collapse routing.  Failures are routed to named local alternatives rather
than repaired by a global estimate.

Audit note: the global critical \(L^3(\mathbb R^3)\) alternatives in the compact
cost criterion remain bookkeeping for the compact-cost route.  They are not used
as hypotheses in the final local no-Type-II theorem.  The terminal wrapper uses
only compact-window critical mass supplied by
`paper6a:lem:terminal-local-critical-mass-routing`.

## Type-I side is separate

This tracker is about the no-Type-II theorem.  Type I closure is not needed to
prove "no Type II singularities."  It is needed only if the manuscript claims
full Navier--Stokes regularity from the final local dichotomy
`paper6:cor:final-local-dichotomy`.

## Discharge criterion for this tracker

The former active targets are discharged by branchwise theorems of the following
form:

> Every Type II singular branch of a suitable weak Navier--Stokes solution,
> after the local renormalizations and ordered selections used in the
> manuscript and after the covered-class entry theorem has been applied,
> satisfies the expanded terminal and ambient adjacent-closure hypotheses, or
> else enters one of the named local alternatives closed by the retired local
> packages below or by the ambient/terminal wrapper itself.

This theorem package is now present in the TeX, so the tracker records no
active obligation for the unconditional no-Type-II goal.

---

# Retired local proof-obligation status inside the covered Type II path

The local proof packages below retire the previously listed assumptions inside
the covered local Type II class.  The entries in this section record discharged
closure statements and audit boundaries.  No item in this section is counted as
active inside the covered path unless one of its audit conditions fails.  These
retirements are now used by the discharged outer wrappers listed above.

## 1. Exhaustiveness of the Type II analytic data

### Current status

Discharged by ordered local routing.  The manuscript now proves the C1
exhaustion interface on the admissible class `\mathcal U_{\mathrm{II}}^{NS}` in
the precise branchwise form used downstream: every admissible branch either has
the Type II analytic data, or its first failed C1 test is assigned to a
class-incompatible alternative or to a named non-retained local state-space
exit.  Consequently, after routed exits are removed, every retained admissible
branch has the Type II analytic data.  The discharge package is:

* `paper6a:lem:typeII-concentration-extraction-routes` — the local
  concentration dichotomy `paper1:thm:paper0-dichotomy` produces a
  Type II concentration point on every admissible branch; empty singular
  set routes to the continuation alternative excluded by clause (iv) of
  `def:c1-admissible-typeII-class`, and the all-Type-I case is excluded
  by clause (iii).
* `paper6a:lem:typeII-scale-alternative-routes` — the supercritical
  Type II scale alternative holds at every concentration point because
  the local Type I alternative is class-incompatible with clause (iii).
* `paper6a:lem:typeII-profile-data-routes` — the local
  scale-translation selection of `paper1:prop:paper0-typeII-reduction`
  produces represented variables and a local Type II sequence
  (`paper1:def:typeII`) under nondegeneracy, supplying the local profile
  datum via `paper1:prop:representation-input`; failure of nondegeneracy
  routes to the multibubble/cascade/gauge-degenerate alternative
  `paper3:def:local-multibubble`(ii)–(v); a nondegenerate
  bounded-selected-window subsequence routes to the scale-rigid
  bounded-window exit `paper3:lem:scale-rigid-bounded-limit-exit`.
* `paper6a:thm:typeII-analytic-data-exhaustive` — combining the three
  lemmas, the Type II analytic data are exhaustive by ordered local routing on
  `\mathcal U_{\mathrm{II}}^{NS}`; each ordered failure of
  `def:c1-exhaustion-failures` is matched to a class-incompatible alternative
  or to a non-retained state-space branch already covered by the local
  stratification, and retained branches have the three analytic data.
* `paper6a:cor:typeII-analytic-data-exhaustiveness-discharged` — the
  ordered-routing exhaustion interface used by
  `thm:c1-typeII-branch-exhaustion` is now a theorem.

No global critical profile theorem and no global \(L^3\) bound are
invoked.  The discharge respects the local state-space stratification:
each failure mode is identified with a named alternative already covered
by the ordered local validation.

At present, no active proof obligation remains inside the covered local Type II
path or in the outer no-Type-II wrapper.  The audit items below should be
reopened only if their local proof packages fail one of the stated boundaries.

## Retired audit note: compact scale-collapse stationary-rigidity input

### Current status

Discharged on the retained local state-space branch.

The manuscript no longer treats compact scale-collapse stationary rigidity as an
independent input to the final multibubble or scale-collapse reductions.  The
TeX now contains:

* `paper5:def:retained-compact-scale-collapse-state`;
* `paper5:lem:retained-scale-collapse-first-failure`;
* `paper5:lem:autonomous-scale-collapse-is-scale-rigid`;
* `paper5:thm:compact-scale-collapse-rigidity-discharged`.

The discharge is local.  It does not prove a new global stationary,
autonomous, or self-similar Liouville theorem.  Instead, the retained thick
autonomous scale-collapse branch is shown to be a scale-rigid local branch and
is eliminated by `paper3:prop:scale-rigidity-discharged`.  If any retained
compact-state input fails, the branch is already assigned to a named local
state-space exit by `paper5:lem:retained-scale-collapse-first-failure`, such as
thin drift, finite scale-cost, modulation failure, compactness failure, rough
core, exterior/separated-core, multicore, or same-point cascade.

`paper8:thm:multibubble-frame-reduction`, the same-point cascade reductions,
and the final assembly now cite the local compact scale-collapse routing
theorem rather than assuming a stationary-rigidity theorem for a nonzero
\(L^3(\mathbb R^3)\) limiting profile.

### Audit checks

The retirement remains valid only if the proof keeps the following boundaries:

* pressure reconstruction, modulation limits, and compactness must be inherited
  from retained local windows, not from a hidden global bound on the original
  solution;
* the stationary omega-limit theorem and Nečas--Růžička--Šverák rigidity remain
  optional shortcuts only, not inputs to the retained-branch exclusion;
* a nonstationary autonomous obstruction must be routed locally into
  scale-rigidity, not discarded by an unstated classification theorem;
* every failure of a retained compact-state input must be assigned to a named
  local state-space alternative by
  `paper5:lem:retained-scale-collapse-first-failure` before the routing theorem
  is invoked;
* same-point cascade uses of the routing theorem must use the local
  active-cascade mass budget, not a global \(L^3(\mathbb R^3)\) bound for the
  original solution.

## Retired audit note: closure of the retained compact tests

### Current status

Discharged.  The manuscript now proves both compact tests on the retained
single-core stratum and routes their failure to named non-retained
alternatives.  The discharge package is:

* `paper6a:lem:retained-tightness-routing` — failure of critical tightness
  routes the branch (via `paper8:thm:exterior-stratification` and
  `paper8:thm:exterior-regularity-no-concentration`) into one of the
  exterior-profile, separated-core, multicore, noncompact-exterior, or
  scale-collapse alternatives, all non-retained.
* `paper6a:lem:retained-h1-routing` — local windowed \(H^1\) control follows
  from `paper7:prop:uniform-windowed-gradient` under the standing inputs
  already supplied by the retained-stratum tests; failure routes to the
  rough-core alternative `paper6:thm:state-decomposition`(iii), which by
  `paper6:thm:rough-core` collapses into the multibubble/cascade alternative.
* `paper6a:thm:retained-compact-tests-closed` — both compact tests are
  theorems on the retained stratum, not assumptions.
* `paper6a:cor:compact-test-closure-discharged` — the obligation is
  discharged on every retained compact single-core branch produced by
  `paper7:lem:no-unrecorded-cost-escape`.

No global tightness or global \(H^1\) estimate is invoked.  The discharge
respects the local state-space stratification: each failure mode is
identified with a named non-retained alternative already covered by the
ordered local validation.


## Items checked but not reopened

The repaired-gauge representation, pressure reconstruction, renormalized local
energy inequality, and Caccioppoli estimates are not reopened here as active
assumptions.  The TeX contains local theorem packages for them, including
`paper2:thm:C`, `paper2:thm:pressure-decomp`, `paper4:ass:lei`,
`paper6a:thm:ac-local-caccioppoli`, and
`paper6a:prop:caccioppoli-regularity-criterion`.  If one of these local inputs
is absent on a branch, the current manuscript routes the absence to an ordered
representation, pressure, rough-core, or state-space alternative rather than
treating it as a hidden global estimate.

The final wrapper theorem `paper6:thm:multibubble` still uses hypothesis
language for terminal admissibility and scale-rigidity.  This is an assembly
audit item, not a separate mathematical assumption, because the TeX now contains
`paper3:prop:scale-rigidity-discharged` and derives terminal admissibility from
the local terminal decoupling package.  The wrapper should ultimately cite those
discharge theorems directly, but reopening it would be warranted only if those
local discharges fail.

---

# Retired audit note: `paper3:ass:scale-rigidity`

## Local scale-rigidity exclusion

### Current status

This item is retired from the active remaining-assumptions list.  The TeX file
now contains an explicit discharge package immediately after
`paper3:ass:scale-rigidity`, culminating in
`paper3:prop:scale-rigidity-discharged`.

The retirement is conditional on an audit of the proof package.  In particular,
the proof must not silently import either of the following as hidden
assumptions:

* a uniform \(L^\infty\) or bounded selected-window limit not derived from the
  stated local compactness, pressure, modulation, and no-subconcentration
  inputs;
* vanishing localized dissipation for the single-core reduction unless it is
  proved for the original selected sequence, not merely for a residual after
  subtracting a fixed limiting profile.

If either point remains unsupported, the item should stay retired only as a
bookkeeping target and the missing boundedness/dissipation lemmas should be
added to a reopened proof-obligation list.

### Audit result

The TeX discharge has been repaired against the two hidden-assumption risks
that motivated this audit:

* `paper3:lem:scale-rigid-pressure-modulation-stability` is now only a
  compactness/stability lemma.  It gives \(L^3\), \(L^\infty_tL^2_x\),
  \(L^2_tH^1_x\), pressure compactness, and modulation stability, but it no
  longer claims \(L^\infty\) boundedness.
* Bounded selected-window limits are now obtained through
  `paper3:lem:scale-rigid-chart-comparability`,
  `paper3:lem:scale-rigid-two-sided-ckn-transfer`,
  `paper3:lem:scale-rigid-ckn-in-gauge`, and
  `paper3:lem:scale-rigid-no-subconcentration-bounded`: chart bounds are
  proved on compact subcylinders before CKN is applied, so boundedness is not
  imported from compactness estimates.
* The represented-to-physical CKN step is now two-sided and local.  It is
  proved by direct chart Jacobians and pressure-oscillation seminorms, not by
  silently reversing the one-way local-transfer estimate.
* Pressure-only CKN concentration is now handled by
  `paper3:lem:scale-rigid-scaled-pressure-decay`,
  `paper3:lem:scale-rigid-dyadic-ckn-persistence`,
  `paper3:lem:scale-rigid-pressure-core-activation`, and
  `paper3:lem:scale-rigid-exterior-pressure-routing`: the decay iteration starts
  from a fixed compact pressure bound, uses \(C_V(2r)\) at the correct outer
  scale, and either activates an \(L^3\) core or routes exterior pressure to the
  separated/exterior structural alternative.
* Newly activated \(L^3\) cores are registered by
  `paper3:lem:scale-rigid-active-core-capture`, so the finite refinement ledger
  cannot hide infinitely many uncounted active objects.
* `paper3:thm:scale-rigid-single-core-compatibility` now verifies the
  single-core branch only when vanishing localized dissipation holds for the
  original selected sequence.  It no longer subtracts a fixed limiting profile
  and treats residual dissipation as original dissipation.
* `paper3:lem:scale-rigid-degenerate-jacobian` no longer infers boundedness
  from gauge degeneracy.  Persistent degeneracy is resolved by the
  no-subconcentration/CKN lemma or by producing a further active core.
* `paper3:prop:scale-rigidity-discharged` no longer invokes
  `paper3:thm:multi`, which assumes `paper3:ass:scale-rigidity`.  The new
  `paper3:lem:scale-rigid-refinement-terminates` uses a finite ledger of
  active objects, pair relations, degenerate selections, and perturbative
  remainders, so refinements cannot cycle without adding a new active profile.

The item remains retired.  Future edits should preserve the audit invariant:
boundedness must come only from chart comparability plus CKN/no-subconcentration,
pressure smallness must come only from the pressure-decay lemma or route to an
exterior structural exit, and single-core dissipation must be dissipation of
the original selected sequence.

### Literal content

This assumption says:

> every **scale-rigid local branch** produced by the same-point or separated-point reductions, including branches where the **relative scale-center Jacobian degenerates**, must fall into one of only two possibilities:
>
> 1. it has a **nonzero bounded selected-window limit**, or
> 2. it **reduces to the single-core branch**;
>
> and therefore such a branch cannot remain a genuine local Type II branch.

So this is not just saying “scale-rigid behavior is restrictive.”
It is saying something much stronger:

> **scale-rigid behavior produces no genuinely new Type II mechanism.**

Either it collapses into something already understood, or it produces a bounded limit object that should already be incompatible with Type II.

---

### What “scale-rigid” means in the logic of the paper

The paper’s multibubble/cascade analysis keeps reducing complicated active profile configurations until only a few structural possibilities remain.

The relevant alternatives are basically:

* a single compact core,
* a multibubble/cascade interaction,
* a perturbative remainder,
* or a **rigid branch** where the relative geometry of scales and centers stops having free dynamics.

So “scale-rigid” is the name for the situation where the geometry of the branch has lost the freedom that would generate genuinely new blowup behavior. Informally, it means:

* the relative scales are no longer producing a rich cascade,
* the relative centers are no longer generating a new branching pattern,
* or the gauge/selection map has become degenerate in a way that is supposed to force reduction rather than new dynamics.

That is why the assumption explicitly includes the case of a **degenerate relative scale-center Jacobian**.
You are folding geometric/gauge degeneracy into the same rigidity bucket.

---

### What this assumption is really trying to do

This assumption is the **closure device** for the entire multibubble reduction.

Earlier in the paper, the same-point and separated-point arguments reduce a complicated active family to three possibilities:

* single-core,
* scale-rigid,
* perturbative.

The perturbative case is supposed to be harmless.
The single-core case is handled elsewhere.
So if you want the multibubble/cascade branch to disappear entirely, you must also kill the scale-rigid branch.

That is exactly what this assumption does.

So the hidden logical form is:

> “All roads that do not already collapse to perturbative behavior or single-core behavior are called scale-rigid, and I assume they are not genuine Type II branches either.”

This is why the assumption is so important.
It is not a minor technical input.
It is one of the **main remaining branch-closure assumptions**.

---

### What the two conclusions inside it mean

## A. “Nonzero bounded selected-window limit”

This means that after passing to suitable terminal windows, the branch converges to a nontrivial bounded limiting object.

Why is that important?

Because a bounded nonzero limit is usually much easier to attack than the original dynamic branch. Once you have such a limit, you try to prove:

* it is autonomous,
* or stationary,
* or smooth,
* or incompatible with the concentration structure,
* or reducible to an already excluded core.

So this part of the assumption is saying:

> scale rigidity forces enough compactness to produce a bounded reduced-limit object.

That is already a serious compactness theorem if proved.

## B. “Reduces to the single-core branch”

This is the other possibility: the scale-rigid branch turns out not to be genuinely multiscale after all.

That means:

* after the geometric/gauge degeneracy is resolved,
* or after collapsing comparable pieces together,
* or after discarding perturbative exterior pieces,

the branch is just a disguised single-core branch.

So this part says:

> rigidity does not generate a new branch class; it collapses back to the single-core case.

---

### Why the assumption is strong

It is strong because it packages several difficult claims into one sentence:

1. **classification claim**
   every scale-rigid branch belongs to one of only two outcomes;

2. **compactness claim**
   one of those outcomes is a bounded selected-window limit;

3. **reduction claim**
   the other outcome reduces to single-core;

4. **exclusion claim**
   either way, no genuine local Type II branch survives.

That is a lot.

The discharge package now shows this by separating the compactness, CKN
regularity, degeneracy, and finite-refinement steps.  The mechanism requires:

* terminal compactness,
* control of modulation parameters,
* control of pressure in the reduced frame,
* and local CKN regularity for no-subconcentration limit objects.

---

### What theorem discharges it

The discharge theorem now has this form:

> Let a local Type II branch satisfy the same-point or separated-point reduction hypotheses and suppose the relative scale-center dynamics are rigid or degenerate. Then, after passing to subsequences and selected windows, either:
>
> * the branch converges to a nonzero bounded reduced limit with the required admissibility properties, or
> * the branch reduces to the nondegenerate compact single-core case.
>   In both cases the branch is excluded.

The implemented proof explicitly bridges:

* geometric rigidity,
* compactness on windows,
* CKN/no-subconcentration boundedness,
* original-sequence single-core dissipation,
* and finite non-circular refinement of structural exits.

The important audit point is that boundedness is not taken from compactness,
and single-core dissipation is not taken from a residual after subtracting a
limit.

---

### How it relates to other assumptions

This one interacts heavily with:

* `paper3:ass:single-core`, because one branch of the dichotomy reduces there;
* `paper3:ass:terminal-admissibility`, because to even analyze the terminal branches you need admissibility;
* the scale-collapse/autonomous-limit machinery in Section 5, because the assumption seems intended to include those exclusions as components.

In other words:

> this assumption is the “rigidity closure” piece of the multibubble argument.

Without it, the multibubble theorem does not fully close.

---

### Discharge package: implemented theorems/lemmas and difficulty

The discharge of `paper3:ass:scale-rigidity` now uses the following results.

**Theorem 1.1: Scale-rigid stratum decomposition.**
Every same-point or separated-point branch whose relative scale-center dynamics
are rigid, including the degenerate relative Jacobian case, enters one of the
local strata used by the discharge: bounded selected-window limit, compact
single-core, perturbative remainder, or a structural exit that is finitely
refined rather than hidden.

**Lemma 1.2: Degenerate Jacobian resolution.**
If the relative scale-center Jacobian degenerates along an active terminal
configuration, then after reselecting the terminal frame the configuration is
assigned to the perturbative, single-core, bounded-limit, or further-active-core
stratum in the sense of Theorem 1.1.  This is a stratification statement, not a
claim that the degenerate case simply disappears.

**Lemma 1.3: Pressure and modulation stability under rigid-frame limits.**
The pressure decomposition, local energy inequality, and repaired-gauge
modulation bounds survive the rigid-frame limiting process strongly enough to
pass to compact selected-window limits, without asserting boundedness.

**Lemma 1.4: No-subconcentration/CKN bounded-window upgrade.**
In a bounded repaired gauge, absence of CKN subconcentration upgrades the
compact limit to a bounded selected-window limit; persistent subconcentration
produces a further active core and exits the residual scale-rigid stratum.

**Lemma 1.5: Bounded selected-window limits exit Type II.**
A nonzero bounded selected-window limit is incompatible with the local Type II
definition used in the multibubble/cascade reduction.

**Theorem 1.6: Single-core reduction compatibility.**
If a scale-rigid branch reduces to a single retained core, then it satisfies the
hypotheses of the compact single-core criterion only when the original selected
sequence has vanishing localized dissipation.

**Lemma 1.7: Finite non-circular structural refinement.**
Structural exits produced inside the scale-rigid proof are resolved by minimal
extraction, same-point/separated reductions, and terminal decoupling; the
iteration terminates by the finite active-profile count and never invokes
`paper3:thm:multi`.

**Difficulty: hard to very hard.**
This is a local state-space stratification theorem, not a standalone regularity
result.  The difficult parts are the degenerate scale-center geometry
and the need to carry pressure/modulation compactness through the local
selected-window reductions.  This is less demanding than an all-branch rigidity
classification because residual rigidity strata may be carried forward instead
of eliminated immediately.

---

# 2. `paper7:def:compact-cost-adjacent-closure`

## Adjacent-stratum closure after the coverage theorem

### Status

The former classwise coverage assumption has been narrowed and discharged.

The manuscript now proves:

* `paper7:thm:classwise-retained-compact-coverage`: every branch with active
  supercritical scaling and validated compact cost divergence either remains in
  the retained compact cost-divergence stratum or enters a named adjacent
  stratum;
* `paper7:cor:retained-stratum-cost-exclusion`: the retained compact stratum is
  excluded.
* `paper7:def:precarrier-validated-compact-channel`: the exact channel reached
  after all non-carrier compact-cost interface tests have passed, but before the
  localized monotonicity carrier is validated.
* `paper7:thm:noncarrier-exits-route`: every first failure before carrier
  validation is routed to an already named exhaustion, representation,
  state-space, critical-mass, compact-test, modulation, cost-well-posedness, or
  cost-compatibility stratum.
* `paper7:thm:precarrier-reduces-to-carrier`: once the branch is in the
  pre-carrier channel, the only remaining compact-cost exits are the carrier
  exits: carrier-subidentity failure, fixed finite error, moving-annulus
  summability, or carrier comparison.
* `paper7:thm:fixed-carrier-closure` and
  `paper7:thm:moving-carrier-closure`: if the corresponding fixed or moving
  localized-energy package is present, the branch is excluded by the local
  cost-to-exclusion theorem.
* `paper7:thm:carrier-package-closes-precarrier`: either fixed or moving
  carrier closure excludes any pre-carrier branch.
* `paper7:lem:precarrier-finite-local-energy`: the finite initial localized
  energy clauses in the carrier packages are supplied after a harmless terminal
  restriction by compact-cylinder local energy times, not by a whole-space
  \(L^3\) bound.
* `paper7:lem:nested-moving-carrier-comparison`: for the canonical identity
  cost, or more generally for any cost energy-controlled by the fixed identity
  cost, nested moving cutoffs give the moving carrier comparison automatically.
* `paper7:thm:carrier-completion-reduced-to-finite-error`: after those
  automatic pieces are removed, carrier completion reduces to finite-error
  carrier selection.
* `paper7:cor:finite-error-selection-gives-carrier-completion`: in the
  canonical or energy-controlled cost channel, finite-error carrier selection
  gives full carrier-estimate completion.
* `paper7:def:carrier-subidentities` and
  `paper7:lem:suitable-energy-gives-carrier-subidentities`: the exact
  localized energy identity is stronger than necessary; the one-sided
  subidentity supplied by suitable local energy is enough.
* `paper7:def:annular-finite-error-selection` and
  `paper7:thm:finite-error-selection-reduced-to-annular`: once suitable local
  energy supplies subidentities, finite-error carrier selection reduces to the
  purely annular finite-error statement.
* `paper7:def:componentwise-annular-finite-error` and
  `paper7:lem:componentwise-gives-annular-finite-error`: the annular statement
  is unpacked into explicit fixed and moving error components: six fixed terms
  and seven moving terms.
* `paper7:def:moving-test-family`,
  `paper7:lem:successful-moving-test-family`,
  `paper7:lem:windowwise-tight-radius-schedule`,
  `paper7:lem:windowwise-shell-pressure-translation`,
  `paper7:cor:windowwise-only-transition-and-scale-remain`,
  `paper7:def:windowwise-transition-routing`,
  `paper7:def:windowwise-positive-scale-routing`,
  `paper7:def:normalized-carrier-blocks`,
  `paper7:lem:closed-carrier-diagnostics`,
  `paper7:lem:carrier-state-compactness-labels`,
  `paper7:lem:nonintegrable-component-certificate`,
  `paper7:lem:transition-carrier-label-is-exterior`,
  `paper7:lem:transition-carrier-state-stratification`,
  `paper7:lem:positive-scale-carrier-core-realization`,
  `paper7:thm:positive-scale-certificate-enters-scale-rigid`,
  `paper7:lem:positive-scale-work-stratification`,
  `paper7:cor:windowwise-transition-routing-holds`,
  `paper7:cor:windowwise-positive-scale-routing-holds`,
  `paper7:thm:windowwise-routing-gives-stratified-routing`, and
  `paper7:cor:adjacent-closure-reduced-to-windowwise-routing`: after
  expanding the moving carrier components, the sharp Section 7 windowwise
  burden is discharged by normalized carrier-state extraction plus the ordered
  local state-space labels.  For monotone radial cutoffs only the
  \(a_+\)-part of the scale-shell term remains,
  `paper7:lem:annular-pressure-flux` reduces the pressure term to enlarged-shell
  \(L^3\) control plus an \(R^{-1}\) tail, and
  `paper7:lem:windowwise-shell-pressure-translation` shows that on the
  canonical windowwise moving family the \(J_2\) through \(J_5\) terms are
  already summable.  The \(J_1\) and \(J_7\) nonintegrable tails are now routed
  by carrier-state stratification; \(J_6\) remains the already assigned
  Section 6 negative-drift branch.
* `paper7:cor:annular-finite-error-selection-gives-carrier-completion`:
  annular finite-error selection gives carrier-estimate completion in the
  canonical or energy-controlled cost channel.
* `paper7:thm:adjacent-closure-reduced-to-carrier-completion`: compact-cost
  adjacent closure follows from non-carrier stratum closure plus
  `paper7:def:carrier-estimate-completion`.

Therefore the Section 7 bottleneck is no longer:

> “Does every compact-cost branch get routed?”

It is only:

> “For each named adjacent exit produced by the ordered tests, is there a local
> exclusion theorem or a route into an already closed state-space alternative?”

That broad residual is now reduced by theorem to the sharper carrier-estimate
completion target, then to annular finite-error selection, then to explicit
componentwise diagnostics, and finally to the sharp windowwise routing
statements that imply the stratified moving-carrier routing package.

---

### Exact reduced statement

For a branch with active supercritical scaling and validated compact Type II cost
divergence, `paper7:lem:no-unrecorded-cost-escape` gives an ordered first-failure
classification.  The branch is either retained or exits through one of the
following finite families:

* analytic/exhaustion failure;
* raw-chart, repaired-gauge, or scale-admissibility failure;
* multibubble, multicore, cascade, or exterior-tightness exit;
* local \(H^1\)-loss;
* finite-cost scale-collapse;
* scale-rigid residual behavior;
* critical-mass failure;
* modulation-integrability failure;
* cost-compatibility failure;
* carrier-subidentity failure;
* fixed-cutoff finite-error failure;
* moving-annulus summability failure;
* carrier-comparison failure for the validated cost.

The compact-cost coverage theorem proves that this list is exhaustive.  The
new reduction theorem proves that all items before the last four carrier rows
are not Section 7 cost-analysis work: they belong to the already named
state-space or interface strata.  The compact-cost-specific residual is the
last four rows.  After the latest reduction, finite initial energy and carrier
comparison are no longer part of the hard residual in the canonical or
energy-controlled channel.  The exact localized energy identity is also no
longer part of the hard residual, because a suitable local energy inequality
supplies the needed one-sided carrier subidentity.  The moving-carrier
failure modes \(J_1\) and \(J_7\) are now routed into already named adjacent
branches by the carrier-state discharge.  On the canonical windowwise moving
family, the \(J_2\) through \(J_5\) terms are already summable, and \(J_6\)
remains assigned to the existing negative-drift branch.

---

### What is already removed from the assumption

The following are no longer open Section 7 assumptions:

* pointwise cost-exclusion data on the retained stratum;
* carrier existence on the retained stratum;
* classwise routing into retained or adjacent strata;
* absence of unrecorded compact-cost escapes.
* reduction of adjacent closure to a finite carrier-estimate target;
* exclusion after the fixed carrier package has been verified;
* exclusion after the moving carrier package has been verified.
* finite localized energy in the carrier package;
* carrier comparison for canonical or nested energy-controlled costs.
* replacement of localized energy identities by suitable-energy subidentities.
* windowwise transition/leakage routing on the canonical moving family.
* windowwise positive-scale routing on the canonical moving family.

Those are now supplied by the local verification proposition, the retained
carrier theorem, the classwise coverage theorem, and the carrier-closure
theorems listed above, together with the carrier-state windowwise routing
corollaries.

---

### Discharged windowwise routing package

The broad adjacent-closure theorem has now been reduced through the canonical
windowwise moving family and the two windowwise routing definitions have been
proved in the TeX:

* `paper7:lem:successful-moving-test-family` closes the branch whenever one
  moving test family has all seven moving components summable.
* `paper7:lem:windowwise-shell-pressure-translation` makes
  \(J_2,J_3,J_4,J_5\) automatic on the canonical family.
* `paper7:def:normalized-carrier-blocks`,
  `paper7:lem:closed-carrier-diagnostics`, and
  `paper7:lem:carrier-state-compactness-labels` construct compact labelled
  carrier states with closed first-label rows;
  `paper7:lem:nonintegrable-component-certificate` handles arbitrary
  nonintegrable tails without assuming persistent unit-window lower bounds.
* `paper7:lem:transition-carrier-label-is-exterior` and
  `paper7:lem:transition-carrier-state-stratification` route \(J_1\)-carrier
  states through the transition-leakage carrier label into the
  exterior/leakage cluster.
* `paper7:lem:positive-scale-carrier-core-realization`,
  `paper7:thm:positive-scale-certificate-enters-scale-rigid`, and
  `paper7:lem:positive-scale-work-stratification` route \(J_7\)-carrier states
  to scale-rigid residual behavior, unless an earlier exterior/leakage,
  modulation-driven recentering, or negative-drift label has already fired.
* `paper7:cor:windowwise-transition-routing-holds` and
  `paper7:cor:windowwise-positive-scale-routing-holds` discharge
  `paper7:def:windowwise-transition-routing` and
  `paper7:def:windowwise-positive-scale-routing`.

The audit point is that this proof must remain carrier-state based.  It must
not reintroduce the false implication that nonintegrability of \(J_1\) or
\(J_7\) gives a uniform pointwise or unit-window lower bound.  Diffuse tails
are routed as normalized local carrier states, and retained \(J_7\)-carrier
states must continue to pass through the compact-core realization lemma before
scale-rigidity is invoked.

---

### How the finite exit list is now organized

The exit list splits into three groups, and the manuscript now records the exact
status of each group.

**Group A: definitional or interface exits.**
Analytic/exhaustion failure, raw-chart failure, repaired-gauge failure,
scale-admissibility failure, modulation-integrability failure, and
cost-compatibility failure do not create a new compact-cost branch.  They are
routed by `paper7:thm:noncarrier-exits-route` to the ordered representation,
repaired-gauge, modulation, and cost-normalization lemmas already cited in
`paper7:lem:no-unrecorded-cost-escape`.

**Group B: structural state-space exits.**
Multibubble, multicore, cascade, exterior-tightness, finite-cost scale-collapse,
scale-rigid residual behavior, and critical-mass failure must be assigned to the
corresponding structural branch arguments.  `paper7:thm:noncarrier-exits-route`
does the assignment.  The genuinely open part here is mostly the existing
scale-rigidity and terminal profile machinery, not Section 7 coverage.

**Group C: carrier-estimate exits.**
Carrier-subidentity failure, fixed-cutoff finite-error failure,
moving-annulus summability failure, and carrier-comparison failure are now the
only exits that still look like Section 7 cost-analysis work.  The manuscript
proves that fixed carrier data imply exclusion and moving carrier data imply
exclusion.  It also proves that finite initial localized energy is automatic and
that carrier comparison is automatic for canonical costs, or for nested moving
cutoffs when the validated cost is energy-controlled by the fixed identity cost.
What remains is therefore not a monolithic PDE theorem.  After replacing exact
identities by suitable-energy subidentities, the moving carrier residual is
read through the explicit component diagnostics in
`paper7:def:componentwise-annular-finite-error`.  In the canonical state-space
reading, \(J_1\) is now routed as a transition carrier state and \(J_7\) is
now routed as a positive scale-work carrier state.  The \(J_2\)--\(J_5\) terms
are already closed on that family, and \(J_6\) remains the already assigned
Section 6 negative-drift branch.  Thus the two windowwise routing definitions
are retired from the active Section 7 assumption list.

---

### Current narrowed bottleneck

There is no active Section 7 windowwise-routing bottleneck left in this
tracker.  The remaining global assembly still uses the already assigned
negative-drift branch, but the separate local cost-divergence exclusion
criterion is also discharged by
`paper6a:thm:cost-divergence-exclusion-discharged` under the same canonical
windowwise routing and adjacent-closure package.  The routing definitions
`paper7:def:windowwise-transition-routing` and
`paper7:def:windowwise-positive-scale-routing` are discharged by theorem.

For the unconditional no-Type-II goal, the ambient applicability of the
adjacent-closure package is now discharged by
`paper7:thm:ambient-adjacent-closure-branchwise` and
`paper7:cor:ambient-adjacent-wrapper-discharged`.  The remaining audit question
is only whether future edits preserve the branchwise local-state-space routing
and avoid replacing it with an unstated global estimate.

---

# 3. `paper6a:ass:cost-divergence-exclusion` -- retired audit item

## Cost-divergence exclusion criterion

This item is no longer counted as an active remaining assumption.

The TeX now contains an explicit discharge package in
`paper6a:subsec:discharge-cost-divergence-assumption`:

* `paper6a:lem:corrected-energy-nonnegative` verifies the nonnegativity of the
  corrected fixed and moving localized energies.
* `paper6a:lem:corrected-monotonicity-versus-divergent-cost` proves the
  arithmetic contradiction: corrected monotonicity, finite initial corrected
  energy, and a nonnegative cost force finite total cost.
* `paper6a:thm:cost-divergence-routing-dichotomy` shows that cost divergence
  on the canonical windowwise moving schedule forces failure of one of the
  remaining local moving-error components \(J_1,J_6,J_7\).
* `paper6a:thm:cost-divergence-exclusion-discharged` routes those failures:
  \(J_1\) enters the transition/exterior route, \(J_7\) enters the
  positive-scale route, and \(J_6\) is the assigned negative-drift route.
* `paper6a:cor:cost-divergence-exclusion-unconditional` records that, under the
  same windowwise routing and adjacent-closure package already used in Section
  7, `paper6a:ass:cost-divergence-exclusion` is a theorem and may be removed
  from `paper6a:def:abstract-final-data` and
  `paper6a:def:expanded-terminal-data`.

The current `type_II_regularity.tex` has been updated accordingly: the final
and expanded terminal hypothesis lists no longer include
`paper6a:ass:cost-divergence-exclusion` as a primitive input.  Later assembly
references now cite the discharge theorem/corollary instead of treating the
criterion as an open assumption.

Audit point: this retirement depends on the canonical windowwise schedule and
the Section 7 adjacent-closure/routing package.  If that package is weakened or
reopened, this item should be rechecked, but it is not an independent active
assumption in the present manuscript.

---

# 4. `paper6a:ass:critical-ns-profile-decomposition` -- retired audit item

## Local terminal state-space completeness

This item is no longer an active global critical-profile assumption.  The TeX
now treats the old label as a local statement and discharges it by
`paper6a:thm:critical-ns-profile-decomposition-discharged`.

The replacement is deliberately not a standard whole-space \(L^3\) profile
decomposition.  It says that on every compact terminal cylinder:

* repaired-gauge, pressure, modulation, local energy, and compact-cylinder
  bounds give local occupation-state compactness;
* positive residual local \(L^3\) mass produces a first closed local
  state-space label;
* the first label is either a named adjacent alternative or the retained
  comparable class;
* adjacent alternatives are closed by their local arguments;
* retained comparable classes are grouped before selecting the terminal frame;
* therefore no positive nonretained residual mass remains on a retained
  terminal branch.

The proof package in `type_II_regularity.tex` is:

* `paper6a:lem:l3-profile-extraction-cited`: compactness of terminal occupation
  states from Aubin--Lions, local pressure reconstruction, and compact-cylinder
  estimates.
* `paper6a:lem:critical-mass-decoupling`: closedness of the ordered local
  state-space labels, now expanded label-by-label in the augmented local
  topology.
* `paper6a:lem:residual-concentration-visible`: selected-time residual
  \(L^3\) concentration is visible either in the full represented branch or in
  the retained comparable component, so cancellation with \(U_n\) cannot hide a
  nonretained residual.
* `paper6a:thm:terminal-adjacent-label-routing`: the adjacent labels are routed
  to already proved local alternatives; terminal completeness no longer assumes
  a separate adjacent-closure hypothesis.
* `paper6a:thm:residual-mass-admissible-state`: positive residual mass first
  produces an augmented local occupation state; Navier--Stokes admissibility is
  applied to the full represented branch, not to the residual as an independent
  equation.  Selected-time residual \(L^3\) concentration is carried by local
  Radon measures, so no time-slice strong-convergence claim is hidden.
* `paper6a:lem:heat-flow-kato-smallness`: positive residual mass produces a
  first closed local state-space label through the augmented-state theorem.
* `paper6a:lem:nonlinear-profile-evolution-cited`: such residual mass cannot
  remain in the retained terminal branch after adjacent closures and comparable
  grouping.
* `paper6a:lem:profile-exhaustion-by-diagonal-extraction`: residual mass is
  exhausted on compact terminal cylinders.
* `paper6a:thm:critical-ns-profile-decomposition-discharged`: the old labelled
  completeness statement follows.
* `paper8:ass:terminal-windowed-profile-completeness`: now a theorem, not an
  assumption, deriving the Section 8 terminal-windowed completeness statement
  from the same local state-space discharge.

The audit point is strict: this retirement is valid only while the proof remains
local.  It must not silently reintroduce:

* a whole-space \(L^3\) profile decomposition;
* a Brézis--Lieb global mass-decoupling identity;
* a global heat/Kato smallness remainder;
* a hidden global boundedness estimate;
* vanishing localized dissipation as an unstated input.

Small residual components are now handled by compact-window product estimates
and local pressure reconstruction, not by global mild-solution theory.  The
terminal pressure decoupling proof uses `paper8:lem:harmonic-cross-pressure-routing`
and nested-ball Calderon--Zygmund reconstruction plus harmonic-tail routing, not
whole-space pressure bounds.
Active components are detected by a CKN-derived local mass floor in a compact
terminal cylinder, with pressure-only detection routed to the pressure/exterior
alternative.

---

# Big-picture comparison of the current proof obligations

After the latest terminal-state pass, the formerly active named assumptions in
the covered local Type II path and the former unconditional-upgrade obligations
have been retired.  Covered-class entry is discharged by
`paper6:thm:physical-typeII-covered-entry` and
`paper6:cor:covered-class-entry-discharged`; ambient adjacent closure is
discharged by `paper7:cor:ambient-adjacent-wrapper-discharged`; and expanded
terminal branchwise applicability is discharged by
`paper6a:cor:expanded-terminal-wrapper-discharged`.  Compact
scale-collapse stationary rigidity, retained compact-test closure, local
scale-rigidity, windowwise routing, cost-divergence exclusion, and local
terminal state-space completeness are retired audit items, not active local
assumptions, unless one of their audit conditions fails.

## A. Retired cost-routing and local cost assumptions

The retained compact Section 7 branch and classwise compact-cost coverage have
been handled.  The broad adjacent-stratum closure statement has been reduced to
carrier-estimate completion plus closure of non-carrier structural/interface
exits, and carrier-estimate completion has been reduced further to
annular finite-error selection in the canonical or energy-controlled cost
channel.  That target has then been unpacked into explicit componentwise
finite-error estimates and finally into the sharp windowwise moving-routing
statements
`paper7:def:windowwise-transition-routing` and
`paper7:def:windowwise-positive-scale-routing`,
with `paper7:lem:successful-moving-test-family` closing the carrier whenever
one moving test family succeeds, `paper7:lem:windowwise-shell-pressure-translation`
making \(J_2\) through \(J_5\) automatic on the canonical family, and the
negative scale-drift term already routed to Section 6.  The two windowwise
routing definitions are now discharged by
`paper7:cor:windowwise-transition-routing-holds` and
`paper7:cor:windowwise-positive-scale-routing-holds`.  The separate final local
cost-divergence criterion used in assembly is then discharged by
`paper6a:thm:cost-divergence-exclusion-discharged` and
`paper6a:cor:cost-divergence-exclusion-unconditional`.

The Section 7 moving-routing discharge and the Section 6A cost-divergence
discharge are retained only as audit trails; neither is counted as an active
assumption.

## B. Retired terminal state-space completeness item

`paper6a:ass:critical-ns-profile-decomposition` is no longer counted as active.
The old global-profile reading has been replaced by compact-window local
state-space completeness, proved in the TeX by
`paper6a:thm:critical-ns-profile-decomposition-discharged` and propagated to
Section 8 by `paper8:ass:terminal-windowed-profile-completeness`.

## Retired audit item

`paper3:ass:scale-rigidity` is no longer counted as active because the
manuscript now contains `paper3:prop:scale-rigidity-discharged`.  The audit
has been tightened in the note above: bounded selected-window limits must come
from CKN/no-subconcentration, and single-core dissipation must be dissipation of
the original selected sequence.

`paper7:def:windowwise-transition-routing` and
`paper7:def:windowwise-positive-scale-routing` are also no longer counted as
active.  Their audit point is that the proof must continue to use normalized
carrier states for diffuse nonintegrability; it must not silently replace
nonintegrable \(J_1\) or \(J_7\) tails by persistent pointwise or unit-window
lower bounds.  The positive-scale proof must also keep the compact-core
realization lemma between a \(J_7\)-carrier state and the invocation of
scale-rigidity.

`paper6a:ass:cost-divergence-exclusion` is also no longer counted as active.
Its audit point is that the discharge must remain tied to the canonical
windowwise moving family, the corrected-monotonicity arithmetic, and the
already assigned \(J_1,J_6,J_7\) routing/adjacent-closure package.

`paper6a:ass:critical-ns-profile-decomposition` is also no longer counted as
active.  Its audit point is that the discharge must remain local: compact
terminal cylinders, local pressure reconstruction, Aubin--Lions occupation
states, closed state-space labels, and local perturbative residual estimates.
It must not reintroduce a global critical profile theorem, global heat/Kato
smallness, hidden global boundedness, or unstated vanishing localized
dissipation.

The compact scale-collapse stationary-rigidity input is also no longer counted
as active.  Its audit point is that the proof must continue to use
`paper5:thm:compact-scale-collapse-rigidity-discharged`: retained thick
autonomous scale collapse is routed into the scale-rigid local branch, while
failure of any retained compact-state input is recorded as a named local exit
by `paper5:lem:retained-scale-collapse-first-failure`.
The stationary omega-limit/NRS argument is only an optional shortcut.

---

# Current status summary

The goal is an unconditional no-Type-II theorem in the local state-space
framework.  No active proof obligation remains for that goal in the current
TeX.  The final branchwise theorem is
`paper6:thm:unconditional-no-typeII`, supported by:

* `paper6:cor:covered-class-entry-discharged`;
* `paper7:cor:ambient-adjacent-wrapper-discharged`;
* `paper6a:cor:expanded-terminal-wrapper-discharged`;
* `paper6a:thm:expanded-terminal-typeII`;
* `paper6:prop:state-elimination`.

The following items are retired audit trails and should be reopened only if
their local proof packages fail one of the stated audit checks.

`paper7:def:windowwise-transition-routing` and
`paper7:def:windowwise-positive-scale-routing` are no longer active: the TeX
now proves them by carrier-state extraction and local state-space
stratification.

`paper6a:ass:cost-divergence-exclusion` is no longer active: the TeX now
derives it from corrected-monotonicity arithmetic plus canonical windowwise
routing and adjacent closure.

`paper6a:ass:critical-ns-profile-decomposition` is no longer active: the TeX now
derives terminal no-hidden-residual completeness from local occupation-state
compactness and the ordered local state-space stratification.

Compact scale-collapse stationary rigidity is no longer active: the TeX now
routes retained compact scale-collapse through
`paper5:thm:compact-scale-collapse-rigidity-discharged`, and the final
multibubble/scale-collapse reductions invoke that local routing theorem rather
than a global Liouville theorem.

Retained compact-test closure is no longer active: the TeX now proves the
critical-tightness and local windowed-\(H^1\) tests on the retained stratum and
routes their failures to named non-retained alternatives.

---

# Audit target for the no-Type-II theorem

The active target list is empty.  The next audit target is consistency: any new
use of the final Type II theorem must cite the full branchwise chain
`paper6:cor:covered-class-entry-discharged`,
`paper7:cor:ambient-adjacent-wrapper-discharged`,
`paper6a:cor:expanded-terminal-wrapper-discharged`, and
`paper6:thm:unconditional-no-typeII`, not only the retired local discharge
packages in isolation.
