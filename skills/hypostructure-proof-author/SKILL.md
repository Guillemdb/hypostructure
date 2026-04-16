---
name: hypostructure-proof-author
description: Author, rewrite, audit, or repair Hypostructure proof objects in docs/source/dataset using docs/source/1_hypostructure_formalism/template.md. Use when Codex must implement the formal template faithfully, verify only the thin interface and local certificate basis, assemble global properties only as theorem outputs, avoid invented K certificates, preserve required closure sections, and prevent the Euler 3D proof mistakes from recurring.
---

# Hypostructure Proof Author

## Purpose

Author proof objects as framework instances, not essays. Every retained statement must instantiate a thin object, interface permit, local node predicate, local certificate, obligation, promotion, dependency cone, local backend package, final certificate chain, theorem-output row, or document-status row.

The invariant is local-to-global:

```text
thin interface + local properties + local backend permits -> K_Goal^+ -> global theorem output
```

Never reverse this arrow. Global regularity, global existence, scattering, structural exclusion, classification, compact attractor existence, and final theorem statements are outputs of the certificate chain, never premises used to certify gates, obligations, Lock tactics, backend permits, or upgrades.

Use `docs/source/1_hypostructure_formalism/template.md` as the source of truth. Use dataset files such as `docs/source/dataset/navier_stokes_3d.md` only for repo-local closure style, not as a substitute for the template.

## Required Workflow

1. Read the target proof file, `docs/source/1_hypostructure_formalism/template.md`, especially `Local-to-Global Certificate Discipline (Mandatory)`, and the relevant framework node/interface docs before editing.
2. Determine the target theorem output and the local certificate basis separately. Do not write the proof until the thin interface, local properties, local backend permits if any, designated goal certificate, and goal dependency cone are explicit.
3. Build or repair the document section-by-section against the template spine. Preserve required sections such as `Formal Proof`, `Part IV`, `Executive Summary: The Proof Dashboard`, `Final Verdict`, and `Document Information` when repo dataset style expects them.
4. Use only framework-native local certificates, anchored local backend certificates, or explicitly declared generic goal certificates derived from local premises. If a `K_...` name is not already defined in the framework or intentionally introduced as part of this proof's public API, do not use it.
5. Keep route facts as route facts. Do not wrap them in synthetic certificate-like objects, do not define `K_{\mathrm{Goal}}` as an alias for such objects, and do not use a global theorem output as a route premise.
6. Emit `K^{inc}` with structured obligations for anything not locally certified. Mark whether each obligation lies in `\Downarrow(K_{\mathrm{Goal}})`.
7. Mirror the current template's `Document Information` schema exactly; include `Local Certificate Basis`, `Global Theorem Output`, `Problem Type`, `Singularity Type`, scoped `Final Status`, and the inc-certificate count when those rows are present.
8. Run a provenance, payload, stale-symbol, local-to-global, and final-status audit before finalizing.

## Hard Guardrails

- Do not hallucinate certificates to make a proof look complete.
- Do not use global theorem statements as input certificates. They may appear only in theorem-output, extraction, final verdict, and document-information rows as consequences of local closure.
- Do not convert backend statements into `K_...` certificates unless the framework or proof explicitly certifies a local backend package or local backend permit.
- Do not overclaim. A status like `UNCONDITIONAL` must be explicitly scoped to the local-certificate-derived designated goal, not to stronger regularity, Lock closure, capacity closure, or global resolution unless those are extracted outputs of that same local chain.
- Do not replace template certificate tables with ad hoc `Item`, `Witness`, or prose tables when the template specifies certificate columns.
- Do not list a vague "backend analytic package" as a certificate. Either name a certified local backend package with provenance or keep backend facts as route data outside the certificate set.
- Do not leave generic template instructions, tutorials, agent notes, or explanatory divagation inside a finalized proof object unless the heading is required and the content has been instantiated as route-local hypostructure data.
- Do not silently skip nodes. Closed-system skips for Nodes 14-16 must be recorded after `K_{\mathrm{Bound}_\partial}^-`.

## Reference Files

Read these references when implementing or reviewing a proof object:

- `references/template-implementation-guide.md`: detailed section-by-section implementation rules, certificate provenance workflow, payload audit, and validation checklist.
- `references/euler-3d-failure-ledger.md`: concrete failure ledger from the Euler 3D proof attempt and exact avoidance rules.

## Minimum Audit Commands

Use targeted searches before finalizing:

```bash
rg -n 'K_\{?\\mathrm|K_\{' docs/source/dataset/<target>.md
rg -n 'Route_|dashboard|replay|step-by-step|machine-checkable|for new problems|TODO|Insert actual|\[TODO' docs/source/dataset/<target>.md
rg -n 'global regularity|global existence|scattering|classification|structural exclusion|compact attractor' docs/source/dataset/<target>.md
rg -n --fixed-strings '<candidate-certificate>' docs/source skills
```

For every `K_...` result, verify provenance and payload shape. For every global-property hit, verify it appears only as a theorem output or final consequence, not as a local premise, missing certificate, upgrade input, Lock hypothesis, or backend permit. For every commentary hit, either remove it or justify it as a required template/style heading with hypostructure-internal content.
