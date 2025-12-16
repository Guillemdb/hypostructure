# The Structural Sieve

## A Certificate-Driven Framework for Singularity Exclusion

Welcome to the documentation for **The Structural Sieve**, a rigorous mathematical framework for analyzing and certifying the behavior of complex dynamical systems, PDEs, and geometric flows.

---

## What is the Structural Sieve?

The Structural Sieve is a **proof-carrying diagnostic algorithm** that transforms the question "does this system develop singularities?" into a systematic verification procedure. Instead of directly proving global regularity, the framework:

1. **Instantiates** the problem through four canonical "thin objects" (Arena, Potential, Cost, Invariance)
2. **Executes** a multi-level diagnostic sieve with 17+ verification nodes
3. **Certifies** the outcome through permit-based proofs
4. **Extracts** quantitative results through Lyapunov reconstruction

---

## Documentation Structure

### [The Framework](./source/hypopermits_jb.md)
Complete specification of the Structural Sieve, including:
- Canonical sieve algorithm
- Interface and barrier registries
- All 17 diagnostic nodes
- Metatheorems and kernel theorems

### [Case Studies](./source/case_studies.md)
Four complete applications demonstrating the framework:
- Poincaré Conjecture (via Ricci flow)
- P vs NP
- Navier-Stokes regularity
- BSD Conjecture

### [Solution Template](./source/template.md)
Step-by-step guide for applying the Sieve to new problems:
- Interface permit checklists
- Execution protocol
- Lyapunov reconstruction procedure

### [System Prompt](./source/system_prompt.md)
Instructions for transforming an LLM into a Hypostructure Diagnostic Engine.

---

## Quick Start

To apply the Structural Sieve to a new problem:

1. **Read** the framework specification to understand the sieve structure
2. **Study** the case studies to see complete execution examples
3. **Follow** the template for your specific problem
4. **Use** the system prompt to engage LLM assistance

---

## Key Concepts

**Thin Objects**: The four foundational structures that encode a problem:
- **Arena** ($\mathcal{A}$): The space where evolution occurs
- **Potential** ($\Phi$): The driving functional
- **Cost** ($\mathcal{C}$): The quantity tracking singularity formation
- **Invariance** ($\mathcal{I}$): The preserved structure

**Permits**: Formal certificates proving properties at each diagnostic node

**Barriers**: Conditions that, when excluded, guarantee regularity

**The Lock**: The final verification stage confirming global behavior

---

## Building This Documentation

```bash
# Install dependencies
make install

# Build the book
make docs

# Serve locally at http://localhost:8000
make serve
```
