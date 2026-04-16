# Hypostructure Proof Template Implementation Guide

## Table of Contents

1. Grounding Pass
2. Template Spine
3. Local-to-Global Certificate Discipline
4. Hypostructure-Only Content Rule
5. Certificate Provenance
6. Goal Certificate Rules
7. Section-by-Section Implementation
8. Payload Contracts
9. Obligation Ledger
10. Final Closure
11. Document Information
12. Validation Checklist

## 1. Grounding Pass

Before writing or patching a proof object:

1. Open `docs/source/1_hypostructure_formalism/template.md`.
2. Open the target dataset file.
3. Open one nearby mature dataset file only for style, usually `docs/source/dataset/navier_stokes_3d.md` for closure layout.
4. Search framework definitions for every certificate family that will be used.
5. Identify the target theorem output and local certificate basis as separate objects.
6. Identify the designated goal certificate and write down what is not being claimed.

Never infer framework semantics from a domain theorem alone. Classical facts can appear as backend route data only if the proof object says they are route data or part of a declared local backend package. They do not automatically become `K_...` certificates.

Grounding must answer these questions before drafting:

- What thin interface is being implemented?
- Which local predicates, permits, node certificates, and local backend permits form `\Gamma_{\mathrm{local}}`?
- Which certificate in the goal cone is actually established from `\Gamma_{\mathrm{local}}`?
- Which global theorem statements are extracted outputs, not prerequisites?

## 2. Template Spine

Use the formal template as the structural source of truth. The proof object should include these conceptual regions in order:

- Metadata
- Label Naming Conventions
- Automation Witness
- Local-to-Global Certificate Discipline
- Abstract
- Theorem Statement
- LLM Execution Protocol dropdown, if the dataset template/style keeps it
- Part 0: Interface Permit Implementation
- Part I: Thin Object Definitions
- Part II: Sieve Execution
- Part II-B: Upgrade Pass
- Part II-C: Breach/Surgery/Re-entry Protocol
- Part III-A: Lyapunov Reconstruction
- Part III-B: Result Extraction / Theorem Output Extraction
- Part III-C: Obligation Ledger
- Formal Proof
- Part IV: Final Certificate Chain
- Executive Summary / Final Verdict, if repo dataset style expects it
- Document Information, if repo dataset style expects it

Do not delete closure sections merely because the user asks for less commentary. Instead, instantiate those sections with terse mathematical route data, certificate tables, and verdict rows.

## 3. Local-to-Global Certificate Discipline

Every proof object must implement the template's arrow:

```text
Gamma_local -> K_Goal^+ -> global theorem output
```

`Gamma_local` may contain only:

- Thin-object data: arena, potential, cost, invariance, singular set, bad set, local state variables.
- Interface permits and node certificates.
- Local backend permits or local backend packages with provenance.
- Lock/local completeness packages if the proof actually uses them.
- Upgrade premises already derived from local certificates.

Global theorem statements are outputs only. They include global regularity, global existence, scattering, structural exclusion, observer-relative censorship, singularity classification, compact attractor existence, final classification, and any named final theorem. They may appear in the theorem statement, Part III-B extraction, Formal Proof conclusion, Part IV, Final Verdict, and Document Information only as consequences of the local chain.

Forbidden placements for global theorem statements:

- `missing` fields of `K^{inc}` payloads.
- Premises of upgrade rules.
- Part 0 required implementations.
- Node certificate payloads.
- Lock hypotheses.
- Backend permit justifications.
- Goal-cone dependency inputs.

If the only way to close a proof is to assume the desired global statement, the correct output is `K^{inc}` or conditional status, not a completed proof.

## 4. Hypostructure-Only Content Rule

Allowed content:

- Definitions of state spaces, dynamics, variables, maps, bad sets, singular sets, and invariants.
- Thin objects: Arena, Potential, Cost, Invariance.
- Permit implementations for Nodes 1-17.
- Local certificates with payloads.
- `inc`, `br`, `blk`, `+`, `-`, and typed diagnostic records.
- Local backend route data clearly labeled as backend route data.
- Promotion rules and non-circularity statements.
- Obligation ledgers and goal-cone membership statements.
- Final certificate chains, theorem-output rows, and verdict rows.

Disallowed content in a finalized proof object:

- Generic tutorial text such as "think like a compiler" unless the template section is intentionally preserved and instantiated.
- Historical context, motivation, dashboards, replay bundles, generic schemas, or implementation advice not specific to the proof.
- Claims that a result is "machine-checkable" unless the certificate chain actually supports that status and repo style requires the row.
- "Approach" and "Result" prose copied from the generic template instead of mathematical route data.
- Any use of global regularity, global existence, scattering, structural exclusion, classification, or final theorem statements as local assumptions.

## 5. Certificate Provenance

For every `K_...` symbol:

1. Search for the exact symbol in `docs/source`.
2. If exact search fails, search the family name, for example `SC_\\lambda`, `Rec_N`, or `CatLib`.
3. If the symbol is a framework-native generic certificate, use the payload shape defined by the node/interface docs.
4. If the symbol is a backend-specific certificate, verify it is declared in the proof object or already established by the repo as a local backend permit/package.
5. If the symbol is neither framework-native nor declared, do not use it. Use plain backend route data, an absent row, or an `inc` diagnostic.

Framework-native examples include generic node certificates such as:

- `K_{D_E}^+`
- `K_{\mathrm{Rec}_N}^+` or `K_{\mathrm{Rec}_N}^{\mathrm{inc}}`
- `K_{C_\mu}^+`
- `K_{\mathrm{SC}_\lambda}^+` or `K_{\mathrm{SC}_\lambda}^-`
- `K_{\mathrm{SC}_{\partial c}}^+`
- `K_{\mathrm{Cap}_H}^{\mathrm{inc}}`
- `K_{\mathrm{LS}_\sigma}^{\mathrm{inc}}`
- `K_{\mathrm{TB}_\pi}^+`
- `K_{\mathrm{TB}_O}^+`
- `K_{\mathrm{TB}_\rho}^{\mathrm{inc}}`
- `K_{\mathrm{RepDesc}_K}^+`
- `K_{\mathrm{GC}_\nabla}^{\mathrm{inc}}`
- `K_{\mathrm{Bound}_\partial}^-`
- `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}`
- `K_{\mathrm{Germ}}^+`, `K_{\mathrm{init}}^+`, `K_{\mathrm{CatLib}}^+` when a Lock completeness package is actually present.

Backend-specific names require stricter proof. Do not create names like `K_{\mathrm{BKM}}^+` or `K_{\mathrm{Euler3DLocal}}^+` just because the classical theorem exists. A backend certificate is acceptable only when its payload is a local property or local continuation permit used by the hypostructure interface; the global theorem produced by that permit remains an output.

## 6. Goal Certificate Rules

The designated goal must be explicit, scoped, and assembled from local certificates. The target theorem output is not part of the local certificate basis.

Valid patterns:

- `K_{\mathrm{Goal}}^+` is declared as the generic designated goal certificate and the proof states exactly which local/interface/backend premise certificates close it.
- `K_{\mathrm{Reg}_{...}}^+` is used only when a certified local continuation or local backend analytic permit has produced it as a promoted certificate.
- `K_{\mathrm{StructReg}_{...}}^+` is used only when the Lock and local completeness package actually establish structural exclusion as an output.
- `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}` is used as a goal only when the proof is specifically a Lock blocking result.

If the generic `K_{\mathrm{Goal}}^+` is used, define its mathematical meaning in the theorem statement, notation table, Part II-B upgrade, Part III-B extraction, Formal Proof, Part IV trace, Final Verdict, and Document Information. Also list the local certificate basis that proves it and the global theorem output extracted from it. A bare `K_{\mathrm{Goal}}^+` with no scoped assertion is not enough.

Invalid patterns:

- `K_{\mathrm{Goal}}^+ := \mathsf{Route}_{...}` where `\mathsf{Route}_{...}` is a newly invented tuple.
- Treating a tuple of backend statements as a proof-algebra certificate.
- Treating global regularity, global existence, scattering, structural exclusion, classification, or final theorem statements as required certificates.
- Claiming global regularity when the goal is only local continuation or singular-route characterization.
- Saying residual obligations are outside the goal cone without showing that the goal does not depend on them.
- Writing "Goal promotion" as a standalone table row without stating the rule, premises, and non-circularity guard.

## 7. Section-by-Section Implementation

### Metadata and Labels

Use the problem slug consistently. Keep theorem labels stable when editing an existing file. Fill `Target Theorem Output` as an output-only global conclusion and `Local Certificate Basis` as the actual proof input.

### Automation Witness

State the type witness and `K_{\mathrm{Auto}}^+` only to the extent the framework supports it. Do not use automation to bypass explicit local backend, Lock, or local continuation certificates. Automation discharges factories, not the global theorem.

### Abstract

Use equations and certificate summaries only:

- State space and dynamics.
- Local certificate basis.
- Designated goal certificate.
- Global theorem output extracted from the local chain.
- Residual diagnostics outside the goal cone.

Avoid "this document presents", "approach", "result", and historical background unless the user explicitly wants prose.

### Theorem Statement

Include:

- Given.
- Claim.
- Local Certificate Basis.
- Designated Goal.
- Theorem Output.
- Notation.

The claim must distinguish local certificates, backend route data, and global outputs. Backend facts are not `K_...` certificates unless locally certified. The global theorem output must never appear in `Given` as a required hypothesis.

### LLM Execution Protocol

If retained, instantiate it. Replace generic algorithm instructions with this proof's route, outcomes, residual diagnostics, upgrade statement, surgery status, and completion criteria.

### Part 0

Instantiate every permit. Do not paste generic implementation templates unless the instance needs them. Part 0 is for local interface implementation only; do not require a global theorem output here. Keep `0.2b` and `0.3b` table schemas faithful:

```markdown
| Certificate | Derived From | Payload | Notes |
```

for `0.2b`, and:

```markdown
| Certificate | Role | Required When |
```

for `0.3b`.

The current template names `0.3b` as local backend/goal certificates. If no optional derived witness certificates are used, write a `none` row rather than inventing certificates.

For `0.3b`, avoid vague rows. If using a local backend analytic package, either:

- Name the package as an actual local certificate with provenance and include it in the certificate algebra; or
- Write it as non-certificate backend route data and keep it out of the final certificate set.

Never list global regularity, global existence, scattering, structural exclusion, final classification, or compact attractor existence as a required `0.3b` certificate.

### Part I

Define only Arena, Potential, Cost, and Invariance. Route facts can be referenced, but do not introduce new certificate names here.

### Part II

Execute nodes in order. Each node should have:

- Question.
- Minimal route-local witness data.
- Certificate.

If a node is unresolved, emit `K^{inc}` with an obligation, missing certificates, failure code, and trace.

### Part II-B

State actual upgrades only. Every upgrade premise must lie in `\Gamma_{\mathrm{local}}`. If the goal closes from local backend route data, state that directly and prove non-circularity. Do not fabricate an arrow from an invented route object to `K_{\mathrm{Goal}}^+`, and do not use the global theorem output as an upgrade premise.

### Part II-C

If no breach, surgery, or re-entry is used, say so. Do not paste generic surgery schemas.

### Part III-A

If no Lyapunov route is used, record null status. Do not create a fake Lyapunov package.

### Part III-B

Extract theorem outputs from the local certificate chain: route data, quantitative bounds, local certificate package, singular-route characterization, functional objects, and actual retroactive upgrades. Keep backend data separate from certificates. Do not move extracted global outputs back into Parts 0-II as inputs.

### Part III-C

Use an obligation table with:

- ID.
- Node.
- Certificate.
- Obligation.
- Missing.
- In Goal Cone?
- Status.

Then state the remaining obligation set and its intersection with the goal cone.

## 8. Payload Contracts

Always check node docs before emitting a native certificate.

Examples:

- `K_{\mathrm{SC}_\lambda}^-` must include `(\alpha,\beta,\lambda_c,\beta-\alpha\ge\lambda_c witness)`.
- `K_{\mathrm{SC}_\lambda}^+` must include `(\alpha,\beta,\lambda_c,\beta-\alpha<\lambda_c proof)`.
- `K_{\mathrm{Rec}_N}^+` should identify the bad set, recovery map/interface, and finite event count when the proof uses the recovery interface.
- `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}` requires an actual Lock tactic and completeness package for structural exclusion.
- `K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}` should record the unresolved Lock obligation and missing tactic or morphism construction.

If the payload cannot be supplied, do not emit `K^+`; emit `K^{inc}`.

## 9. Obligation Ledger

Every `inc` or `br-inc` certificate creates an obligation. The ledger must state whether the obligation lies in `\Downarrow(K_{\mathrm{Goal}})`.

Rules:

- Residual obligations may remain only outside the designated goal cone.
- If an obligation is inside the goal cone, either discharge it or downgrade the final status.
- Do not call the whole mathematical problem solved if only the designated local/backend route is certified and the requested global theorem was not extracted by a named promotion/extraction theorem.
- The final verdict must name the exact certificate and scope.

## 10. Final Closure

The final closure should include:

- Formal Proof.
- Part IV validity checklist.
- Core node trace.
- Local backend and goal trace.
- Obligation ledger summary.
- Local certificate basis and global theorem output summary.
- Executive Summary / Final Verdict, if used in dataset style.
- Document Information, if used in dataset style.

The final proof should derive the goal from local certificate context and local backend route data without using the goal or global theorem output as a premise.

## 11. Document Information

Mirror the current template's `Document Information` table. At minimum, when the table is present, include:

- `Document Type`.
- `Framework`.
- `Problem Class`.
- `Local Certificate Basis`.
- `Global Theorem Output`.
- `Problem Type`.
- `System Type`.
- `Singularity Type`.
- `Verification Level`.
- `Inc Certificates`.
- `Final Status`.
- `Generated`.

Rules:

- `Problem Type` must be a hypostructure class, not a narrative description.
- `Singularity Type` must be a singularity-table class or `REGULAR`; if unresolved, explicitly say the unresolved sector is auxiliary.
- `Verification Level` must not overclaim. If the proof object was not actually machine-checked, use the repo's required wording only if it means template-level proof-object status, not external formal verification.
- `Local Certificate Basis` must list proof inputs, not global conclusions.
- `Global Theorem Output` must list consequences extracted from the local basis, not assumptions.
- `Final Status` must be scoped, for example `UNCONDITIONAL PROOF FROM LOCAL CERTIFICATES for K_{\mathrm{Goal}}^+`, `CONDITIONAL`, `HORIZON`, or `GOAL NOT REACHED`.

## 12. Validation Checklist

Run these checks before finalizing:

1. Section check: all required template sections or repo-style closure sections are present.
2. Certificate provenance check: every `K_...` symbol is framework-native, repo-anchored, or explicitly declared.
3. Payload check: every native certificate has required fields.
4. Goal check: no `K_{\mathrm{Goal}} :=` alias to a synthetic route object.
5. Obligation check: every `inc` / `br-inc` appears in the ledger and has goal-cone status.
6. Commentary check: no generic template instructions, tutorials, replay bundles, or unrelated dashboards remain.
7. Scope check: final status is scoped to the exact designated goal.
8. Local-to-global check: every proof input is local/interface/backend-local, and every global property appears only as an extracted theorem output.
9. Output-as-input check: global regularity, global existence, scattering, structural exclusion, classification, compact attractor existence, and final theorem statements do not appear in `missing`, upgrade premises, Part 0 requirements, node certificates, Lock hypotheses, or backend permit justifications.
10. Document information check: table rows match the current template schema and do not use stale fields.
11. Stale-symbol check: every removed or rejected certificate/route symbol has zero remaining uses outside the failure ledger.
12. Build/parse check: run the docs build if feasible, or at least inspect MyST directives and tables.
