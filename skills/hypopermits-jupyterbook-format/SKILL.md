---
name: hypopermits-jupyterbook-format
description: "Normalize or convert Markdown documents to the Jupyter Book/MyST format used by docs/source/hypopermits_jb.md, including prf: directives, admonitions for remarks, internal/external hyperlinks, and LaTeX math conventions (especially table math without raw |). Use when updating project docs to match that format."
---

# Hypopermits JupyterBook Format

## Overview

Normalize a Markdown document to the Jupyter Book style in `docs/source/hypopermits_jb.md`, preserving content while standardizing proof directives, admonitions, cross-references, and LaTeX math.

## Workflow

1. Open `docs/source/hypopermits_jb.md` and the target document; scan the section structure, directive usage, labels, and admonitions to mirror.
2. Align scaffolding: keep YAML frontmatter (title/subtitle/author) and heading hierarchy consistent with the reference document.
3. Label section headers using MyST target syntax on the line above the heading (e.g., `(sec-some-section)=`), mirroring `docs/source/hypopermits_jb.md`; avoid inline `{#sec-...}` heading anchors (as seen in `docs/source/sketches/fragile/fragile-index.md`).
4. Convert formal statements to `prf:` directives (`definition`, `theorem`, `lemma`, `corollary`, `axiom`, `proof`, `example`, `remark`); add `:label:` using the same naming style (`def-`, `thm-`, `lem-`, `cor-`, `ax-`, `rem-`, `proof-`, `sec-`) but **never** use numeric patterns (numbers are generated dynamically elsewhere). Use descriptive slugs from titles/keywords instead, and keep any `:class:` annotations.
5. Wrap remarks and callouts in admonitions: use `:::{prf:remark}` for remarks and `:::{note}`, `:::{tip}`, `:::{warning}` for other callouts; include a `:label:` line for every admonition (use `rem-` for `prf:remark`, `note-`/`tip-`/`warn-` for the others) and use `:class: dropdown` for long tips.
6. Replace informal references with links: use `{prf:ref}` for proof directives, `{ref}` for sections/labels, `{cite}` for citations, and `[text](url)` for external URLs.
7. Normalize math: use `$...$` for inline and `$$...$$` for display; use `\mathcal`, `\mathbb`, `\mathbf`, `\mathrm`, and `\text{}`; use `\text{-}` for hyphenated math terms (e.g., `\infty\text{-topos}`).
8. Fix table math: remove raw `|` inside table cells and replace with `\lvert...\rvert`, `\left|...\right|`, `\lVert...\rVert`, or `\mid`, then wrap in `$...$`.

## Conventions Reference

Use `skills/hypopermits-jupyterbook-format/references/hypopermits_jb_conventions.md` for concrete syntax patterns and examples.

## Guardrails

- Preserve meaning; change only formatting and LaTeX conventions unless a correction is required for math clarity.
- Keep labels stable when migrating existing references; add new labels following the same prefix style without numeric patterns.
