# Euler 3D Failure Ledger

## Table of Contents

1. Purpose
2. Failures and Avoidance Rules
3. Bad Certificate Names
4. Review Checklist for Future Proofs

## 1. Purpose

This ledger records the concrete mistakes made while rewriting `docs/source/dataset/euler_3d.md`. Use it as a mandatory anti-pattern checklist when creating or repairing any future hypostructure proof object.

## 2. Failures and Avoidance Rules

### Failure 1: Treating "only hypostructure math" as permission to delete required closure structure

What went wrong:

- Removed or proposed removing closure material such as `Formal Proof`, final verdict, and document information.
- Over-prioritized minimalism over faithful dataset/template implementation.

Avoidance rule:

- Preserve required template and repo-style closure sections. Rewrite their contents as hypostructure math instead of deleting the sections.

### Failure 2: Confusing the raw template with the fully implemented dataset style

What went wrong:

- Initially used only the raw template spine and missed that `navier_stokes_3d.md` implements additional closure expectations such as final verdict and document information.

Avoidance rule:

- Use `template.md` for semantics and required slots.
- Use mature dataset files only for local presentation closure.
- If they differ, do not invent a hybrid; keep template semantics and add only repo-style closure sections that express actual certificate data.

### Failure 3: Leaving generic instructional commentary in proof slots

What went wrong:

- Kept or reintroduced headings and language such as execution protocol, dashboard, machine-checkable status, and agent instructions without always reducing them to instance-local route mathematics.

Avoidance rule:

- If a template heading is retained, instantiate it with route, permit, certificate, obligation, and verdict data only.
- Remove generic tutorial text, prior context, replay framing, and process narration.

### Failure 4: Treating unanchored certificate names as valid because they sounded useful

What went wrong:

- Used or preserved Euler-specific names with no repo-wide or framework anchor.
- Presented classical backend theorems as if they were hypostructure certificates.

Avoidance rule:

- Search exact `K_...` symbols before using them.
- If the symbol is not framework-native or explicitly declared by a backend package, do not use it.
- Represent classical facts as backend route data unless a certificate package is actually supplied.

### Failure 5: Hallucinating optional witness certificates

What went wrong:

- Promoted helicity conservation and flow-map representation into certificate-like objects.

Avoidance rule:

- Optional witness data can be recorded as route data.
- Do not give it a `K_...` name unless the framework defines that certificate or the proof explicitly declares a derived witness certificate with template-shaped provenance.

### Failure 6: Using the wrong typed negative certificate form

What went wrong:

- Used `K_{\mathrm{SC}_\lambda}^{\mathrm{wit}}`, which was not the framework-native node certificate.

Avoidance rule:

- For scaling, use the node contract:
  `K_{\mathrm{SC}_\lambda}^+=(\alpha,\beta,\lambda_c,\beta-\alpha<\lambda_c proof)`.
  `K_{\mathrm{SC}_\lambda}^-=(\alpha,\beta,\lambda_c,\beta-\alpha\ge\lambda_c witness)`.

### Failure 7: Under-specifying native certificate payloads

What went wrong:

- Emitted `K_{\mathrm{SC}_\lambda}^-` with only `\alpha=-1` and prose.
- Emitted `K_{\mathrm{Rec}_N}^+` without enough recovery-interface data.

Avoidance rule:

- Read the node/interface definition before emitting a certificate.
- Include all required payload fields.
- If the payload is unavailable, emit `K^{inc}` rather than a partial `K^+`.

### Failure 8: Defining the goal certificate as an invented route object

What went wrong:

- Defined `K_{\mathrm{Goal}}^+:=(\mathsf{Route}_{\mathrm{Euler3D}})`.
- Used the invented `\mathsf{Route}_{\mathrm{Euler3D}}` tuple as a proof-algebra premise.

Avoidance rule:

- Never define `K_{\mathrm{Goal}}` as an alias to an invented tuple.
- Keep backend route data as equations and route statements.
- Let the goal be an actual designated certificate, and state how the certified route data close it.

### Failure 9: Breaking template table semantics

What went wrong:

- Changed `0.2b Derived Witness Certificates` into a `Witness | Status | Payload` table.
- Changed `0.3b Goal and Local Backend Certificates` into an `Item | Status | Role` table.

Avoidance rule:

- Preserve template-shaped tables:
  `Certificate | Derived From | Payload | Notes` for `0.2b`.
  `Certificate | Role | Required When` for `0.3b`.
- If there is no certificate, write a `none` row rather than changing the schema.

### Failure 10: Overstating final status

What went wrong:

- Used language that could be read as solving stronger Euler 3D regularity or Lock closure claims.

Avoidance rule:

- State final status only for the designated goal.
- Explicitly list capacity, stiffness, mixing, oscillation, and Lock residuals as outside the goal cone if they remain unresolved.
- Never let `UNCONDITIONAL` stand without its scope.

### Failure 11: Not auditing after each patch

What went wrong:

- Left a stale `\mathsf{Route}_{\mathrm{Euler3D}}` reference after removing the synthetic route object.

Avoidance rule:

- After every certificate or goal refactor, run literal searches for removed symbols.
- Treat any stale symbol as a blocker before finalizing.

### Failure 12: Allowing "backend package" language to become vague

What went wrong:

- Replaced hallucinated certificates with "backend package" language without always making clear that the package is route data, not a new certificate.
- Left a `Backend analytic package` row that could be read as a certificate although no named local package certificate had been proven.

Avoidance rule:

- If a local backend analytic package is a certificate, name and declare it.
- If it is not a certificate, call it backend route data and keep it out of the certificate set.

### Failure 13: Leaving the generic goal certificate too bare after removing the bad alias

What went wrong:

- Removed `K_{\mathrm{Goal}}^+:=(\mathsf{Route}_{\mathrm{Euler3D}})` but still relied on a bare `K_{\mathrm{Goal}}^+` in several places.
- Did not immediately force every theorem, notation, upgrade, proof, final trace, and verdict row to restate the exact mathematical content and scope of the generic goal.

Avoidance rule:

- If `K_{\mathrm{Goal}}^+` is generic, every major closure section must specify what it certifies.
- The goal must be scoped as a local-certificate-derived continuation/stretching route, not as a global regularity premise, Lock closure premise, or capacity closure premise.

### Failure 14: Treating "goal-cone empty" as a slogan instead of an audited dependency statement

What went wrong:

- Wrote residual obligations as outside `\Downarrow(K_{\mathrm{Goal}}^+)` without enough local dependency evidence in every closure table.
- Let `GOAL-CONE EMPTY` appear in final-status rows without pairing it with the residual-obligation table and the non-circularity guard.

Avoidance rule:

- Every residual diagnostic must have an obligation ID, missing certificates, goal-cone status, and reason it is not used by the designated goal.
- `GOAL-CONE EMPTY` is allowed only if Part III-C and Part IV show the exact residual set and its empty intersection with the goal cone.

### Failure 15: Using a stale or incomplete Document Information schema

What went wrong:

- The closure initially missed `Document Information`.
- After restoring it, the table risked drifting from the current template fields, especially `Problem Type`, `Singularity Type`, and scoped `Final Status`.

Avoidance rule:

- Mirror the current `template.md` `Document Information` table.
- Include `Local Certificate Basis`, `Global Theorem Output`, `Problem Type`, `Singularity Type`, scoped `Final Status`, inc-certificate counts, and generated date.
- Do not reuse older dataset schemas if the template has changed since the last commit.

### Failure 16: Treating dashboard/final-verdict style as license for non-mathematical commentary

What went wrong:

- Restored `Executive Summary: The Proof Dashboard` and `Final Verdict` because repo style expected them, but these sections can easily become prose commentary.

Avoidance rule:

- Dashboard and verdict sections must be tables or terse rows of hypostructure data: system objects, node outcomes, Lock status, final certificate, residual obligations, and exact scope.
- Do not add explanatory narrative, motivational summaries, or "proof dashboard" claims that are not backed by certificates.

### Failure 17: Using outcome labels that do not match the emitted certificate type

What went wrong:

- The execution protocol used generic `K_X^{\mathrm{wit}}` language while the actual scaling node was repaired to the framework-native `K_{\mathrm{SC}_\lambda}^-`.

Avoidance rule:

- The execution-protocol outcome table must match the actual emitted certificates.
- If the framework defines a native negative certificate such as `K_{\mathrm{SC}_\lambda}^-`, use that exact type consistently in the node trace and summary.

### Failure 18: Not checking validation commands themselves

What went wrong:

- An audit command pattern was initially over-escaped and unsuitable as a copy-paste guard.

Avoidance rule:

- Any command included in a skill or proof-maintenance guide must be tested or at least syntactically checked in the repo shell.
- Broken validation commands are a proof-maintenance failure because future agents will skip or mis-run the audit.

### Failure 19: Letting global theorem outputs leak into certificate inputs

What went wrong:

- The skill and template language still centered "goal/backend package" before explicitly separating the local certificate basis from the global theorem output.
- Wording around analytic regularity and backend packages could be read as requiring global regularity, scattering, structural exclusion, or final classification as proof inputs.
- The old review checklist did not force an output-as-input audit.

Avoidance rule:

- Start every proof with the template arrow:
  `thin interface + local properties + local backend permits -> K_{\mathrm{Goal}}^+ -> global theorem output`.
- Verify only thin-object data, interface permits, local node predicates, local backend permits, and local completeness packages.
- Treat global regularity, global existence, scattering, structural exclusion, singularity classification, compact attractor existence, and final theorem statements as extracted outputs only.
- Never put a global theorem output in `missing`, upgrade premises, Part 0 required implementations, node payloads, Lock hypotheses, backend permit justifications, or the local certificate basis.

## 3. Bad Certificate Names

Do not use these names unless they are later formally defined by the framework or an explicit backend package:

- `K_{\mathrm{Euler3DLocal}}^+`
- `K_{\mathrm{VortEq3D}}^+`
- `K_{\mathrm{BiotSavart3D}}^+`
- `K_{\mathrm{BKM}}^+`
- `K_{\mathrm{StretchRoute}}^+`
- `K_{\mathrm{SingRoute}_{\mathrm{Euler3D}}}^+`
- `K_{\mathrm{Helicity}}^+`
- `K_{\mathrm{FlowMap}}^+`
- `K_{\mathrm{SC}_\lambda}^{\mathrm{wit}}`

Also avoid invented route wrappers such as:

- `\mathsf{Route}_{\mathrm{Euler3D}}` when used as a certificate premise or goal alias.

Plain route tuples are allowed only as non-certificate exposition if the user explicitly wants that notation and it is never placed in the certificate algebra.

## 4. Review Checklist for Future Proofs

Before finalizing a new hypostructure proof:

1. Search for every `K_...` name and verify provenance.
2. Search for every route shorthand and verify it is not used as a certificate.
3. Check `0.2b` and `0.3b` table columns against the template.
4. Check `K_{\mathrm{SC}_\lambda}^{\pm}` payloads against the node definition.
5. Check `K_{\mathrm{Rec}_N}^{+}` payload has bad set, recovery map/interface, and event count.
6. Check every `inc` and `br-inc` has an obligation ledger entry.
7. Check final verdict is scoped to the designated goal.
8. Check unresolved diagnostics are explicitly outside `\Downarrow(K_{\mathrm{Goal}})`.
9. Check no stale invented names remain after refactors.
10. Check that generic commentary was either removed or converted into proof-object data.
11. Check `Document Information` matches the current template schema.
12. Check generic `K_{\mathrm{Goal}}^+` is scoped everywhere it appears.
13. Check dashboard and verdict sections contain certificate data, not explanatory prose.
14. Check any audit commands in the skill or proof notes run without regex or shell errors.
15. Check the local-to-global arrow: thin interface and local properties are inputs; global properties appear only as outputs.
16. Check that no global theorem statement appears in `missing`, upgrade premises, Part 0 requirements, node certificates, Lock hypotheses, backend permit justifications, or the local certificate basis.
