# Hypopermits JupyterBook Formatting Conventions

Use `docs/source/hypopermits_jb.md` as the canonical example. Mirror its syntax, spacing, and label style.

## Frontmatter and Headings

```yaml
---
title: "Document Title"
subtitle: "Optional Subtitle"
author: "Author Name"
---
```

Follow with a top-level `#` title line and the same heading hierarchy (`##`, `###`, etc.) used in the reference.

## Section Labels

Use MyST targets on a separate line above the heading:

```md
(sec-some-section)=
## 2. Some Section Title
```

Use the `sec-` prefix with descriptive slugs (no numeric patterns). Avoid inline heading anchors like `{#sec-some-section}`.

## Proof and Math Directives

```md
:::{prf:definition} Title
:label: def-some-concept

Definition text...
:::
```

Use the same directive style for `theorem`, `lemma`, `corollary`, `axiom`, `proof`, `example`, and `remark`. Add `:class:` where needed. Labels should be descriptive slugs (no numeric patterns); numbers are generated dynamically elsewhere.

Theorem titles sometimes include a short code in brackets:

```md
:::{prf:theorem} [KRNL-Example] Descriptive Title
:label: thm-example
```

When nesting directives or admonitions, increase the colon fence depth (`::::`) to avoid clashes and close with a matching fence.

## Admonitions

```md
:::{prf:remark} Remark Title
:label: rem-some-remark

Remark text...
:::
```

Use `:::{note}`, `:::{tip}`, `:::{warning}` for non-proof callouts. Add a `:label:` line for each admonition (`note-`, `tip-`, `warn-` prefixes) and add `:class: dropdown` for long tips or notes.

## Cross-References and Links

- Use `{prf:ref}` to reference proof directives: `({prf:ref}`def-some-concept`)`.
- Use `{ref}` to reference section or standard labels: `Section {ref}`sec-overview``.
- Use `{cite}` for citations: `{cite}`AuthorYear``.
- Use Markdown links for external URLs: `[text](https://example.com)`.

## LaTeX Conventions

- Inline math uses `$...$`; display math uses `$$...$$`.
- Use `\mathcal`, `\mathbb`, `\mathbf`, `\mathrm`, and `\text{}` for consistent typography.
- Use `\text{-}` for hyphenation inside math (e.g., `\infty\text{-topos}`).
- Use `\to`, `\mapsto`, `\Rightarrow`, `\iff` for arrows and logical implication.
- Use `\left|...\right|`, `\lvert...\rvert`, `\lVert...\rVert` for absolute values and norms.
- Use `\mid` for conditional or divisibility bars.

## Table Math Rules

In tables, pipes (`|`) are column separators only. Avoid raw `|` inside math cells. Replace with `\lvert...\rvert`, `\left|...\right|`, `\lVert...\rVert`, or `\mid`, and wrap math in `$...$`.
