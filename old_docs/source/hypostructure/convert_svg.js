#!/usr/bin/env node
/**
 * Convert SVG to PDF using Puppeteer
 * Required because the SVG uses foreignObject elements with HTML content
 * that Inkscape cannot render.
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function convertSvgToPdf(svgPath, pdfPath) {
    const absoluteSvgPath = path.resolve(svgPath);
    const absolutePdfPath = path.resolve(pdfPath);

    // Read SVG to get dimensions
    const svgContent = fs.readFileSync(absoluteSvgPath, 'utf8');

    // Extract viewBox dimensions
    const viewBoxMatch = svgContent.match(/viewBox="([^"]+)"/);
    let width = 2519;
    let height = 9852;

    if (viewBoxMatch) {
        const [, , , vbWidth, vbHeight] = viewBoxMatch[1].split(/\s+/).map(Number);
        if (vbWidth && vbHeight) {
            width = vbWidth;
            height = vbHeight;
        }
    }

    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    const page = await browser.newPage();

    // Set viewport to match SVG dimensions
    await page.setViewport({
        width: Math.ceil(width),
        height: Math.ceil(height),
        deviceScaleFactor: 2  // Higher resolution
    });

    // Navigate to SVG file
    await page.goto(`file://${absoluteSvgPath}`, {
        waitUntil: 'networkidle0'
    });

    // Wait for fonts to load
    await page.evaluate(() => document.fonts.ready);

    // Export to PDF
    await page.pdf({
        path: absolutePdfPath,
        width: `${width}px`,
        height: `${height}px`,
        printBackground: true,
        margin: { top: 0, right: 0, bottom: 0, left: 0 }
    });

    await browser.close();
    console.log(`Converted ${svgPath} to ${pdfPath}`);
}

// CLI usage
const args = process.argv.slice(2);
if (args.length < 2) {
    console.error('Usage: node convert_svg.js <input.svg> <output.pdf>');
    process.exit(1);
}

convertSvgToPdf(args[0], args[1]).catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
