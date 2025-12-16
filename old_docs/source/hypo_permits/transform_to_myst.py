#!/usr/bin/env python3
"""
Transform LaTeX-style markdown to MyST Jupyter Book format.
Converts theorem environments, lists, formatting, and references.
"""

import re
import sys

def transform_yaml_frontmatter(content):
    """Simplify YAML frontmatter for Jupyter Book."""
    # Match the YAML frontmatter
    yaml_pattern = r'^---\n(.*?)\n---'
    match = re.search(yaml_pattern, content, re.DOTALL)

    if match:
        new_frontmatter = '''---
title: "The Structural Sieve: A Certificate-Driven Framework for Singularity Exclusion"
subtitle: "Operational Semantics and Permit Vocabulary for the Hypostructure Diagnostic Engine"
author: "Guillem Duran Ballester"
---'''
        content = re.sub(yaml_pattern, new_frontmatter, content, flags=re.DOTALL)

    return content

def transform_theorem_environments(content):
    """Convert LaTeX theorem environments to MyST prf directives."""

    # Map of environment types to prf directive types
    env_map = {
        'definition': 'definition',
        'theorem': 'theorem',
        'lemma': 'lemma',
        'proof': 'proof',
        'remark': 'remark',
        'corollary': 'corollary',
        'proposition': 'proposition',
        'metatheorem': 'theorem',  # Will add :class: metatheorem
    }

    # Pattern for environments with title and label
    # \begin{env}[Title]\label{xxx}
    pattern_with_title_label = r'\\begin\{(' + '|'.join(env_map.keys()) + r')\}\[([^\]]+)\]\\label\{([^}]+)\}'

    def replace_with_title_label(match):
        env_type = match.group(1)
        title = match.group(2)
        label = match.group(3).replace(':', '-')  # Convert colons to hyphens
        prf_type = env_map[env_type]

        if env_type == 'metatheorem':
            return f':::{{prf:{prf_type}}} {title}\n:label: {label}\n:class: metatheorem\n'
        else:
            return f':::{{prf:{prf_type}}} {title}\n:label: {label}\n'

    content = re.sub(pattern_with_title_label, replace_with_title_label, content)

    # Pattern for environments with only label (no title)
    # \begin{env}\label{xxx}
    pattern_with_label = r'\\begin\{(' + '|'.join(env_map.keys()) + r')\}\\label\{([^}]+)\}'

    def replace_with_label(match):
        env_type = match.group(1)
        label = match.group(2).replace(':', '-')
        prf_type = env_map[env_type]

        if env_type == 'metatheorem':
            return f':::{{prf:{prf_type}}}\n:label: {label}\n:class: metatheorem\n'
        else:
            return f':::{{prf:{prf_type}}}\n:label: {label}\n'

    content = re.sub(pattern_with_label, replace_with_label, content)

    # Pattern for proof (no title, no label)
    # \begin{proof}
    content = re.sub(r'\\begin\{proof\}', ':::{prf:proof}\n', content)

    # Pattern for end of environments
    for env in env_map.keys():
        content = re.sub(r'\\end\{' + env + r'\}', ':::', content)

    return content

def transform_lists(content):
    """Convert LaTeX list environments to markdown lists."""

    # We need to track whether we're in enumerate or itemize
    # and convert \item accordingly

    lines = content.split('\n')
    result = []
    list_stack = []  # Stack to track nested lists ('enumerate' or 'itemize')
    item_counters = []  # Counter for enumerate

    for line in lines:
        stripped = line.strip()

        # Check for begin enumerate
        if stripped == r'\begin{enumerate}':
            list_stack.append('enumerate')
            item_counters.append(0)
            continue

        # Check for begin itemize
        if stripped == r'\begin{itemize}':
            list_stack.append('itemize')
            item_counters.append(0)
            continue

        # Check for end enumerate or itemize
        if stripped == r'\end{enumerate}' or stripped == r'\end{itemize}':
            if list_stack:
                list_stack.pop()
                item_counters.pop()
            continue

        # Check for \item
        if stripped.startswith(r'\item'):
            if list_stack:
                list_type = list_stack[-1]
                indent = '   ' * (len(list_stack) - 1)
                item_content = stripped[5:].strip()  # Remove \item

                if list_type == 'enumerate':
                    item_counters[-1] += 1
                    result.append(f'{indent}{item_counters[-1]}. {item_content}')
                else:
                    result.append(f'{indent}- {item_content}')
            else:
                # \item outside of list context, keep as is
                result.append(line)
            continue

        result.append(line)

    return '\n'.join(result)

def transform_text_formatting(content):
    """Convert LaTeX text formatting to markdown."""

    # \textbf{text} -> **text**
    content = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', content)

    # \textit{text} -> *text*
    content = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', content)

    # \texttt{text} -> `text`
    content = re.sub(r'\\texttt\{([^}]+)\}', r'`\1`', content)

    return content

def transform_references(content):
    """Convert LaTeX references to MyST format."""

    # Theorem~\ref{xxx} or Definition~\ref{xxx} -> {prf:ref}`xxx`
    # The ~ is a non-breaking space in LaTeX
    content = re.sub(r'(Theorem|Definition|Lemma|Remark|Corollary|Proposition|Metatheorem)~?\\ref\{([^}]+)\}',
                     lambda m: f'{{prf:ref}}`{m.group(2).replace(":", "-")}`', content)

    # Generic \ref{xxx} -> {ref}`xxx`
    content = re.sub(r'\\ref\{([^}]+)\}',
                     lambda m: f'{{ref}}`{m.group(1).replace(":", "-")}`', content)

    return content

def transform_tables(content):
    """Convert LaTeX tabular environments to markdown tables."""

    # Find tabular environments
    tabular_pattern = r'\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}'

    def convert_table(match):
        table_content = match.group(1)

        # Split by \\ (row separator)
        rows = re.split(r'\\\\', table_content)

        md_rows = []
        header_done = False

        for row in rows:
            row = row.strip()
            if not row or row == r'\hline':
                continue

            # Split by & (column separator)
            cells = [c.strip() for c in row.split('&')]

            # Clean up cells - remove \hline if present
            cells = [c.replace(r'\hline', '').strip() for c in cells]

            if cells and any(c for c in cells):
                md_row = '| ' + ' | '.join(cells) + ' |'
                md_rows.append(md_row)

                # Add separator after first row (header)
                if not header_done:
                    separator = '|' + '|'.join(['---' for _ in cells]) + '|'
                    md_rows.append(separator)
                    header_done = True

        return '\n'.join(md_rows)

    content = re.sub(tabular_pattern, convert_table, content, flags=re.DOTALL)

    # Remove \begin{center} and \end{center}
    content = re.sub(r'\\begin\{center\}\s*', '', content)
    content = re.sub(r'\s*\\end\{center\}', '', content)

    return content

def transform_math(content):
    """Handle math environments - mostly keep as-is but fix align."""

    # Convert \begin{align} to $$\begin{aligned}
    content = re.sub(r'\\begin\{align\}', r'$$\\begin{aligned}', content)
    content = re.sub(r'\\end\{align\}', r'\\end{aligned}$$', content)

    return content

def cleanup(content):
    """Final cleanup passes."""

    # Remove any remaining \label{} commands (they should be in directives now)
    # Be careful not to remove labels inside math

    # Fix multiple consecutive blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Ensure ::: closes have blank line before them for proper parsing
    # But don't add if already there
    content = re.sub(r'([^\n])\n:::', r'\1\n\n:::', content)

    return content

def main():
    input_file = '/home/guillem/hypostructure/docs/source/hypo_permits/hypopermits.md'
    output_file = '/home/guillem/hypostructure/docs/source/hypo_permits/hypopermits_jb.md'

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply transformations in order
    print("Transforming YAML frontmatter...")
    content = transform_yaml_frontmatter(content)

    print("Transforming theorem environments...")
    content = transform_theorem_environments(content)

    print("Transforming lists...")
    content = transform_lists(content)

    print("Transforming text formatting...")
    content = transform_text_formatting(content)

    print("Transforming references...")
    content = transform_references(content)

    print("Transforming tables...")
    content = transform_tables(content)

    print("Transforming math...")
    content = transform_math(content)

    print("Cleaning up...")
    content = cleanup(content)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Done! Output written to {output_file}")

if __name__ == '__main__':
    main()
