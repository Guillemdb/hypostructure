#!/usr/bin/env python3
"""Render the Algebraic Geometry diagram HTML to a single-page PDF using Playwright."""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

async def render_to_pdf():
    html_path = Path(__file__).parent / "ag_diagram_standalone.html"
    svg_path = Path(__file__).parent / "ag_diagram.svg"
    png_path = Path(__file__).parent / "ag_diagram.png"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        # Set a very large viewport to capture the full diagram
        page = await browser.new_page(viewport={"width": 2000, "height": 20000})

        # Load the HTML file
        await page.goto(f"file://{html_path.absolute()}")

        # Wait for Mermaid to render
        await page.wait_for_timeout(4000)

        # Extract SVG and save it
        svg_content = await page.evaluate('''() => {
            const svg = document.querySelector('.mermaid svg');
            if (svg) {
                return svg.outerHTML;
            }
            return null;
        }''')

        if svg_content:
            with open(svg_path, 'w') as f:
                f.write(svg_content)
            print(f"SVG saved to: {svg_path}")

        # Get the mermaid element bounds
        bounds = await page.evaluate('''() => {
            const el = document.querySelector('.mermaid');
            if (el) {
                const rect = el.getBoundingClientRect();
                return { x: rect.x, y: rect.y, width: rect.width, height: rect.height };
            }
            return null;
        }''')

        if bounds:
            await page.screenshot(
                path=str(png_path),
                clip={
                    "x": bounds["x"],
                    "y": bounds["y"],
                    "width": bounds["width"],
                    "height": bounds["height"]
                },
                type="png"
            )
            print(f"PNG saved to: {png_path}")
            print(f"Dimensions: {bounds['width']}x{bounds['height']}")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(render_to_pdf())
