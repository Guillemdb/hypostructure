#!/usr/bin/env python3
"""Render mermaid diagram from markdown to PDF using Playwright."""

import sys
import re
from pathlib import Path
from playwright.sync_api import sync_playwright


def extract_mermaid(md_path: str) -> str:
    """Extract mermaid code block from markdown file."""
    content = Path(md_path).read_text()
    match = re.search(r'```mermaid\n(.*?)```', content, re.DOTALL)
    if not match:
        raise ValueError(f"No mermaid code block found in {md_path}")
    return match.group(1)


def render_to_pdf(mermaid_code: str, output_path: str):
    """Render mermaid code to PDF using Playwright."""
    html = f'''<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{ margin: 0; padding: 20px; background: white; }}
    </style>
</head>
<body>
    <div class="mermaid">{mermaid_code}</div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'default', flowchart: {{ useMaxWidth: false }} }});
    </script>
</body>
</html>'''

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html)
        page.wait_for_selector('.mermaid svg', timeout=60000)

        svg = page.query_selector('.mermaid svg')
        bbox = svg.bounding_box()

        page.set_viewport_size({
            "width": int(bbox['width']) + 40,
            "height": int(bbox['height']) + 40
        })

        page.pdf(
            path=output_path,
            print_background=True,
            width=f"{int(bbox['width']) + 40}px",
            height=f"{int(bbox['height']) + 40}px"
        )
        browser.close()


if __name__ == '__main__':
    md_file = sys.argv[1] if len(sys.argv) > 1 else 'diagram_framework.md'
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else 'diagram_framework.pdf'

    mermaid_code = extract_mermaid(md_file)
    render_to_pdf(mermaid_code, pdf_file)
    print(f"Generated: {pdf_file}")
