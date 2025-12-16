#!/usr/bin/env python3
"""
Clean Mermaid-generated SVG for arXiv compatibility.
Removes embedded fonts but keeps styling CSS for text rendering.
"""

import re
import sys
from pathlib import Path

def clean_svg(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    # Remove only the first <style> block that contains the embedded font
    # This matches from the OFL license comment through the @font-face declaration
    svg_content = re.sub(
        r'<style[^>]*>/\* Copyright 2019 The Recursive Project.*?</style>',
        '',
        svg_content,
        count=1,
        flags=re.DOTALL
    )

    # Replace "Recursive Variable" font references with standard fonts
    svg_content = re.sub(
        r'"Recursive Variable"',
        '"Arial", "Helvetica"',
        svg_content
    )

    # Also fix font-family in style attributes
    svg_content = re.sub(
        r"'Recursive Variable'",
        "'Arial', 'Helvetica'",
        svg_content
    )

    # Remove xmlns:xhtml namespace declaration if present (can cause issues)
    svg_content = re.sub(r'\s*xmlns:xhtml="[^"]*"', '', svg_content)

    # Write the cleaned SVG
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"Cleaned SVG written to {output_path}")

    # Print size comparison
    input_size = Path(input_path).stat().st_size
    output_size = Path(output_path).stat().st_size
    print(f"Size reduction: {input_size:,} -> {output_size:,} bytes ({100*(1-output_size/input_size):.1f}% smaller)")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.svg> <output.svg>")
        sys.exit(1)

    clean_svg(sys.argv[1], sys.argv[2])
