# Remaining assumptions in the current Type II manuscript

This tracker records the assumptions and named proof obligations that are still
open in the current `type_II_regularity.tex` manuscript.

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
  separately, because the manuscript treats it as replaceable by the terminal
  critical profile package based on
  `paper6a:ass:critical-ns-profile-decomposition`.
* `paper6a:ass:cost-divergence-exclusion` is now discharged by
  `paper6a:thm:cost-divergence-exclusion-discharged` and
  `paper6a:cor:cost-divergence-exclusion-unconditional`; it is retained below
  only as an audit item, not as an active remaining assumption.

Each active remaining item is read in the same way:

1. what it literally says,
2. what it means conceptually,
3. what role it plays in the argument,
4. what kind of theorem would be needed to discharge it,
5. how strong it is,
6. how it relates to the other assumptions.

Retired items may still be recorded below as audit notes, but they are not
counted as open assumptions unless the audit fails and the item is reopened.

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
added to the active proof-obligation list.

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
  energy clauses in the carrier packages are automatic from the positive finite
  critical \(L^3\) annulus.
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

# 4. `paper6a:ass:critical-ns-profile-decomposition`

## Critical Navier–Stokes profile decomposition

This is the most technically rich assumption on your list.

---

### Literal content

It assumes that for any bounded sequence \(u_{0,n}\) in
\(L^3_\sigma(\mathbb R^3)\), or in a stronger admissible critical space, one
can pass to a subsequence and write
\[
u_{0,n}
=
\sum_{j=1}^J \Lambda_{\lambda_{j,n},x_{j,n}}\phi^j + r_n^J,
\qquad
(\Lambda_{\lambda,x_0}f)(x)
=
\lambda^{-1}f\left(\frac{x-x_0}{\lambda}\right),
\]

with:

* profiles (\phi^j),
* scales (\lambda_{j,n}),
* centers (x_{j,n}),
* remainder (r_n^J),

and with several additional properties.

This is much more than just “a decomposition exists.”

---

## Clause A: pairwise orthogonality of parameters

This is the standard profile-decomposition geometry:
different profiles are separated either by

* scales,
* centers,
* or both.

That means they do not asymptotically overlap in a way that would destroy decoupling.

### Why it matters

Without orthogonality, the decomposition is not really isolating independent concentration mechanisms.

It is the geometric backbone of the whole decomposition.

---

## Clause B: critical mass decoupling

You assume
\[
\|u_{0,n}\|_{L^3}^3
=
\sum_{j=1}^J \|\phi^j\|_{L^3}^3
+
\|r_n^J\|_{L^3}^3
+
o_n(1).
\]

This is the critical (L^3) decoupling identity.

### Why it matters

This lets you say:

* only finitely many profiles can carry norm above a fixed threshold,
* small profiles are really small in the critical sense,
* and the total critical mass is exhausted by the extracted profiles plus remainder.

This is essential for separating:

* active profiles,
* from perturbative profiles.

---

## Clause C: perturbatively small remainder after linear evolution

You assume
[
\lim_{J\to\infty}\limsup_{n\to\infty}
|e^{t\Delta}r_n^J|*{X*{\rm Kato}}=0.
]

This says the linear heat evolution of the remainder is small in a critical Kato space.

### Why it matters

This is the analytic engine that lets you treat the remainder as perturbative after all large profiles are extracted.

Without this, the remainder could still hide serious nonlinear behavior.

---

## Clause D: nonlinear remainder is perturbatively small on compact profile-time windows

This is the nonlinear version of the previous clause.

### Why it matters

Your paper is not doing purely linear profile decomposition.
It wants to analyze actual Navier–Stokes dynamics.

So you need more than linear smallness:
you need the remainder to stay negligible under nonlinear evolution on the windows relevant to profile analysis.

That is what lets you say:

* once the active profiles are removed,
* the rest is harmless.

---

## Clause E: the final completeness clause

This is the special part.

It says:

> if, after removing all extracted profiles, some terminal coordinate frame still contains nonzero local critical mass, then the profile theorem must produce an additional nonzero profile associated to that frame.

This is not just standard decomposition.
This is a **no-hidden-mass** principle tailored to your terminal analysis.

---

### Why the final clause is so important

Your terminal arguments need to know that the profile decomposition is **complete relative to the terminal frames actually used in the Type II analysis**.

Otherwise a dangerous loophole remains:

> maybe after removing all extracted profiles, there is still nontrivial local critical mass in some terminal frame, but it never gets represented by an extracted profile.

If that loophole exists, your terminal profile analysis is incomplete.

This final clause closes that loophole.

So this assumption is doing two jobs at once:

1. standard critical NS profile decomposition,
2. a terminal-frame completeness theorem.

That second job is the more specialized one.

---

### What this assumption is used for in the paper

It is used to prove terminal critical local compactness:

* all non-scattering critical profiles appear as active profiles,
* small profiles are perturbative by Kato theory,
* and there is no hidden terminal mass left in the remainder.

That is exactly how the paper tries to justify:

* terminal profile completeness,
* and the claim that the residual component is perturbatively small on compact profile-time windows.

So this assumption is one of the main engines behind the terminal completion of the profile analysis.

---

### Why it is strong

Because it combines:

* standard profile decomposition,
* nonlinear perturbation theory,
* critical mass decoupling,
* and a custom “hidden terminal mass implies another profile” clause.

The first several parts resemble known critical-profile technology.
The last part is stronger and more tailored to your manuscript.

So as a single assumption, it is very powerful.

---

### What part is standard and what part is nonstandard

This distinction is important.

## Standard-looking part

These parts are in the spirit of known critical NS profile theory:

* bounded critical sequence,
* extraction of orthogonal profiles,
* norm decoupling,
* perturbatively small remainder after linear evolution,
* nonlinear stability for small profiles.

That portion reads like external theory you would normally cite.

## Nonstandard / paper-specific part

The final clause:

* if terminal-frame mass remains, extract another profile tied to that frame.

That is not just standard textbook profile decomposition wording.
It is a manuscript-specific completeness clause adapted to your terminal Type II geometry.

So if you were to rewrite this assumption later, the natural split would be:

* cite the standard profile theorem as a theorem,
* isolate the terminal completeness clause as the genuinely extra input.

---

### What proving it would require

To fully discharge the whole assumption as written, you would need:

1. a standard critical Navier–Stokes profile decomposition theorem;
2. a perturbative nonlinear profile theorem in the chosen critical topology;
3. a proof that the terminal coordinate-frame selection used in your paper is compatible with profile extraction in such a way that any hidden local mass actually yields another profile.

That third step is the manuscript-specific difficulty.

---

### Why it matters so much

Because many of your terminal arguments depend on the claim that after extracting the active profiles, the remainder is genuinely harmless.

Without this assumption, the remainder might still contain:

* hidden critical mass,
* hidden active concentration,
* or frame-dependent terminal activity that the analysis never sees.

So this assumption is the main anti-hidden-mass principle of the terminal part of the paper.

---

### Discharge package: required theorems/lemmas and difficulty

To discharge `paper6a:ass:critical-ns-profile-decomposition`, the paper should
separate the standard critical profile theorem from the terminal-frame
completeness theorem.

**Theorem 4.1: Critical \(L^3\) profile decomposition.**
Every bounded sequence in the chosen critical divergence-free space admits an
orthogonal scale-center profile decomposition with \(L^3\) mass decoupling.

**Lemma 4.2: Brezis--Lieb decoupling in the selected critical topology.**
The profile expansion gives the required critical norm decoupling for the
velocity sequence and for localized terminal annuli used in the paper.

**Theorem 4.3: Linear Kato-small remainder.**
After extracting all profiles, the heat evolution of the remainder is small in
the critical Kato topology needed for Navier--Stokes perturbation theory.

**Theorem 4.4: Nonlinear profile decomposition and stability.**
The nonlinear Navier--Stokes evolutions of the extracted profiles decouple on
compact profile-time windows, and the nonlinear remainder remains perturbative
after all active profiles are removed.

**Lemma 4.5: Terminal-frame compatibility.**
The terminal coordinate frames selected by the Type II analysis are either
asymptotically orthogonal to the extracted profiles or coincide with one of the
profile frames after passing to a subsequence.

**Theorem 4.6: No hidden terminal mass.**
If a terminal coordinate frame still carries nonzero local critical mass after
subtracting all extracted profiles, then the profile decomposition can be
continued to produce an additional nonzero profile attached to that frame.

**Difficulty: very hard.**
The standard profile-decomposition and perturbative stability pieces are
known-type hard results.  The terminal-frame compatibility and no-hidden-mass
clauses are paper-specific and likely require a new argument tying critical
profile extraction to the exact terminal geometry used by the Type II proof.

---

# Big-picture comparison of the active remaining proof obligations

After retiring `paper3:ass:scale-rigidity` and
`paper6a:ass:cost-divergence-exclusion` to audit-only status, the active
remaining proof obligations have narrowed to the terminal profile-completion
package.

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

## B. Active terminal profile-completion assumption

This says the terminal critical profile analysis is genuinely complete:

* `paper6a:ass:critical-ns-profile-decomposition`

This is the anti-hidden-mass engine.

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

---

# My blunt summary of what each active one is really asking for

`paper7:def:windowwise-transition-routing` and
`paper7:def:windowwise-positive-scale-routing` are no longer active: the TeX
now proves them by carrier-state extraction and local state-space
stratification.

`paper6a:ass:cost-divergence-exclusion` is no longer active: the TeX now
derives it from corrected-monotonicity arithmetic plus canonical windowwise
routing and adjacent closure.

`paper6a:ass:critical-ns-profile-decomposition`
asks for a theorem saying **critical profile extraction is complete and leaves no hidden terminal mass**.

---

# Difficulty order, easiest to hardest

Only one active item remains in this tracker.

1. **`paper6a:ass:critical-ns-profile-decomposition` — very hard.**
   The standard critical profile decomposition and perturbative stability are
   known-type hard results, but the terminal-frame compatibility and no-hidden
   terminal mass clauses are manuscript-specific and likely require new work.
   This remains the deepest item because hidden terminal mass cannot simply be
   routed away; the profile machinery must actually detect it.
