# Reusable Hypostructure Framework Layer

Date: 2026-04-16

## Purpose

This document separates the hypostructure-specific proof machinery from a
particular backend such as Burgers 1D. A new proof backend should reuse the same
certificate route, trace verification, Lock package accounting, upgrade
accounting, and final target wrapper without copying Burgers-specific code.

## Implemented Reusable Pieces

The reusable framework layer now includes:

- Hypostructure.Framework.Certificates: generic positive, no-witness, blocked,
  breached, benign, pathological, and stagnation certificate wrappers, plus the
  generic payload families used by gates and barriers.
- Hypostructure.Framework.Execution: generic execution traces, trace locations,
  outcome tags, upgrade rules, Lock certificate packages, obligation ledgers,
  final certificate chains, and RunValidity.
- Hypostructure.Framework.Route: the new reusable route verifier. A backend
  supplies a TraceBackedRoute and a TraceBackedRouteProof; the framework
  produces RunValidity.meetsTemplateCompletionCriteria.
- Hypostructure.Framework.Upgrade: reusable E1-E13 Lock tactic identifiers,
  Lock tactic dossiers, local/bridge/Lock-to-target upgrade dossiers, and a
  combined CertifiedUpgradeDossier. This is the reusable accounting layer for
  the analytic upgrade without baking in any PDE-specific theorem.
- Hypostructure.Framework.Rigor: a lightweight classification of proof
  boundaries into framework-proved logic (`F`), reusable literature facts (`L`),
  and problem-specific math (`P`). Backends can expose their open boundary with
  `RigorBoundaryItem` records instead of mixing all axioms together.
- Hypostructure.Framework.Upgrade also contains the reusable a-posteriori
  upgrade metatheorems: `APosterioriLocalization.target` for a uniform
  localization theorem over all objects, and `APosterioriLocalization.point`
  for the local-estimates pattern where a backend builds a fresh local
  certificate package for the single object currently being proved. A backend
  provides an admissibility predicate, a desired witness predicate, a
  bad-morphism proposition, a localization theorem from missing witness to bad
  morphism, and a Lock proof excluding that bad morphism. The framework then
  proves the target witness. `APosterioriLocalizationDossier` packages the
  uniform pattern for reuse across backends.
- TraceBackedTargetClaim: a generic final wrapper saying that the final
  certificate is recorded, the hypostructure route is valid, and the semantic
  target theorem has been proved.
- Hypostructure.Sieve.FiniteDag: reusable finite DAG execution and closure
  machinery.
- Hypostructure.Sieve.GenericProgram: the current 0-truncated generic sieve
  execution example over NodeTag.

The Burgers route now instantiates TraceBackedRoute instead of defining its own
route-validity record by hand. It also instantiates the reusable upgrade layer:
the Lock is recorded as an E5 functional tactic dossier, and the final
regularity theorem is exposed through CertifiedUpgradeDossier.

## E1-E13 vs Analytic Upgrade

E1-E13 are reusable Lock tactics. They are used to prove that a bad morphism or
bad obstruction route is blocked, for example the Burgers E5 functional tactic
uses Cole-Hopf plus heat certificates to block the represented bad germ.

The analytic upgrade is downstream of that Lock. Its reusable framework parts
are now twofold: certificate accounting via `LocalToTargetUpgradeDossier`, and
the generic a-posteriori contrapositive theorem via
`APosterioriLocalization.target` or `APosterioriLocalization.point`. The
pointwise form is the preferred shape for global results from local estimates:
the backend builds local certificates for the current admissible object, the
Lock blocks the corresponding bad morphism, and the framework returns the
target witness for that same object. The localization theorem itself remains
problem-specific: for each backend, it must prove that a missing semantic
witness produces a bad morphism in the Lock target class. For Burgers, this is
the remaining a-posteriori PDE compactness/continuation argument.

## What A New Backend Must Provide

A new proof backend should not reimplement the framework route machinery. It
must provide only the problem-specific semantics and local certificate proofs:

- A concrete state carrier and, when needed, a solution or evolution carrier.
- Concrete semantic predicates: initial condition, residual or equation,
  boundary discipline, local regularity, uniqueness, and final target theorem.
- Local certificate data proving the framework payload meanings for the problem:
  energy, compactness, capacity, stiffness, topology, representation, boundary,
  and any problem-specific bridge certificates.
- A finite execution trace whose certificate names match the template route.
- A TraceBackedRoute listing required trace certificates, Lock certificates,
  analytic certificates, preservation certificates, upgrade rules, obligation
  ledger, and final certificate chain.
- A TraceBackedRouteProof proving that the route lists are actually present in
  the trace and final chain and that the goal cone is empty.
- A LockTacticDossier selecting one of E1-E13 and proving the corresponding
  backend obstruction.
- A LocalToTargetUpgradeDossier recording the local certificates, bridge
  certificates, Lock certificate, target certificate, and non-circularity of
  the final promotion.
- A CertifiedUpgradeDossier combining the selected E1-E13 Lock tactic with the
  final local-to-target upgrade.
- When applicable, either an APosterioriLocalizationDossier for a uniform
  missing-witness-to-bad-morphism localization theorem or a pointwise local
  estimate provider using `APosterioriLocalization.point` to derive the
  semantic target one admissible object at a time.
- Backend analytic theorems that instantiate the reusable a-posteriori theorem
  or otherwise turn the local certificates plus Lock and bridge packages into
  the semantic target theorem.

## What Is Still Backend-Specific

These pieces must remain outside the reusable framework:

- PDE identities, weak residuals, and solution semantics.
- Fourier, semigroup, and Cole-Hopf constructions for Burgers.
- Problem-specific compactness libraries and representation dictionaries.
- Problem-specific Lock obstruction definitions.
- The final localization theorem for each mathematical problem: the proof that
  failure of the desired witness actually emits a route-local bad morphism
  covered by that problem's certified library.

For Burgers 1D, the remaining nonconstant work is still the heat semigroup on
the full PeriodicH1State carrier, the genuine nonconstant Cole-Hopf bridge,
nonconstant compactness/representation certificate factories, and the
a-posteriori localization theorem from missing regularity witness to a
route-local finite-time H1 bad morphism.

## Literature Modules

Reusable mathematical facts that are not hypostructure logic now live outside
the Burgers backend. Current literature modules are:

- Hypostructure.Literature.Heat.Periodic1D: standard periodic 1D heat
  existence, uniqueness, smoothing, and contraction package, currently exposed
  as `periodicHeat1D_semigroupBackend_literature` until formalized from
  mathlib/Fourier analysis.
- Hypostructure.Literature.ColeHopf.PeriodicBurgers1D: standard periodic
  Cole-Hopf transform and residual/uniqueness bridge for Burgers-like PDEs,
  currently exposed as `periodicBurgers1D_coleHopfBackend_literature` until the
  transform is constructed directly.

These are `L`-layer facts: they are reusable analytic infrastructure, not
framework theorems and not one-off Burgers proof hacks.

## Acceptance Criteria For Reuse

A new backend is using the reusable hypostructure layer correctly when:

- It imports Hypostructure.Framework.Route instead of copying route-validity
  structures.
- It imports Hypostructure.Framework.Upgrade instead of inventing backend-local
  copies of E1-E13, Lock tactic dossiers, or final upgrade dossiers.
- Its final proof exposes a TraceBackedTargetClaim or a domain-specific claim
  assembled from TraceBackedRouteProof.targetClaimOfCertifiedUpgrade.
- Its route validity is proved by TraceBackedRouteProof.runValidity_holds.
- All problem-specific hard math appears in local certificate theorems or the
  final analytic upgrade, not in the framework layer.
- The framework layer contains no references to the backend state type,
  equation, or target theorem.
