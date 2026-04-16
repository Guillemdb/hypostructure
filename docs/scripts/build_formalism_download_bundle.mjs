import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = path.resolve(SCRIPT_DIR, '..');
const MYST_CONFIG = path.join(DOCS_DIR, 'myst.yml');
const OUTPUT_FILE = path.join(DOCS_DIR, '_static', 'hypostructure-formalism-bundle.js');
const BUILDER_FILE = path.join(DOCS_DIR, '_static', 'hypostructure-prompt-builder.html');
const WIDGET_CSS_FILE = path.join(DOCS_DIR, '_static', 'hypostructure-download-widget.css');
const WIDGET_JS_FILE = path.join(DOCS_DIR, '_static', 'hypostructure-download-widget.js');

const FORMALISM_ENTRY = 'source/1_hypostructure_formalism/intro_hypostructure.md';
const TEMPLATE_ENTRY = 'source/1_hypostructure_formalism/template.md';
const DATASET_ENTRY = 'source/dataset/dataset.md';
const DATASET_DIR = path.join(DOCS_DIR, 'source', 'dataset');
const DEFAULT_DATASET_SELECTION = new Set([
  'source/dataset/burgers_1d.md',
  'source/dataset/navier_stokes_3d.md',
]);

function indentOf(line) {
  return line.match(/^\s*/)[0].length;
}

function cleanYamlValue(value) {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function findFollowingTitle(lines, startIndex, itemIndent) {
  for (let index = startIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (index > startIndex + 1 && indentOf(line) <= itemIndent && trimmed.startsWith('- ')) {
      return undefined;
    }

    const match = trimmed.match(/^title:\s+(.+)$/);
    if (match) return cleanYamlValue(match[1]);
  }

  return undefined;
}

function parseTocSection(configText, rootFile, filePattern, defaultTitle) {
  const lines = configText.split(/\r?\n/);
  const rootIndex = lines.findIndex((line) => line.trim() === `- file: ${rootFile}`);

  if (rootIndex === -1) {
    throw new Error(`Could not find TOC entry: ${rootFile}`);
  }

  const rootIndent = indentOf(lines[rootIndex]);
  const entries = [
    {
      file: rootFile,
      title: findFollowingTitle(lines, rootIndex, rootIndent) || defaultTitle,
    },
  ];

  for (let index = rootIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();
    const indent = indentOf(line);

    if (indent <= rootIndent && trimmed.startsWith('- file:')) break;

    const match = trimmed.match(/^- file:\s+(\S+\.md)$/);
    if (!match) continue;

    const file = match[1];
    if (file === rootFile || !filePattern.test(file)) continue;

    entries.push({
      file,
      title: findFollowingTitle(lines, index, indent) || titleFromPath(file),
    });
  }

  return entries;
}

function parseFormalismToc(configText) {
  return parseTocSection(
    configText,
    FORMALISM_ENTRY,
    /^source\/1_hypostructure_formalism\/\S+\.md$/,
    'Hypostructure Formalism',
  );
}

function parseDatasetToc(configText) {
  return parseTocSection(
    configText,
    DATASET_ENTRY,
    /^source\/dataset\/\S+\.md$/,
    'Dataset',
  );
}

function parseFrontmatterTitle(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) return undefined;

  const titleMatch = match[1].match(/^title:\s+(.+)$/m);
  return titleMatch ? cleanYamlValue(titleMatch[1]) : undefined;
}

function titleFromPath(filePath) {
  return path
    .basename(filePath, '.md')
    .replace(/^\d+[a-z]?_/, '')
    .replace(/[-_]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function idFromPath(filePath, kind) {
  return filePath
    .replace(/^source\/1_hypostructure_formalism\//, '')
    .replace(/^source\/dataset\//, '')
    .replace(/\.md$/, '')
    .toLowerCase()
    .replace(/^/, `${kind}-`)
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function readEntry(entry) {
  const absolutePath = path.join(DOCS_DIR, entry.file);

  if (!fs.existsSync(absolutePath)) {
    throw new Error(`Missing source file: ${entry.file}`);
  }

  const content = fs.readFileSync(absolutePath, 'utf8').replace(/\r\n/g, '\n');

  return {
    id: idFromPath(entry.file, entry.kind),
    kind: entry.kind,
    title: entry.title || parseFrontmatterTitle(content) || titleFromPath(entry.file),
    path: entry.file,
    defaultSelected: entry.defaultSelected,
    content,
  };
}

function appendTemplate(entries) {
  if (entries.some((entry) => entry.file === TEMPLATE_ENTRY)) return entries;

  const content = fs.readFileSync(path.join(DOCS_DIR, TEMPLATE_ENTRY), 'utf8');
  return [
    ...entries,
    {
      file: TEMPLATE_ENTRY,
      kind: 'formalism',
      title: parseFrontmatterTitle(content) || 'Hypostructure Proof Object Template',
      defaultSelected: true,
    },
  ];
}

function withDefaults(entries, kind, defaultSelected) {
  return entries.map((entry) => ({ ...entry, kind, defaultSelected }));
}

function appendDatasetDirectoryFiles(entries) {
  const seen = new Set(entries.map((entry) => entry.file));
  const extraEntries = fs.readdirSync(DATASET_DIR)
    .filter((name) => name.endsWith('.md'))
    .map((name) => `source/dataset/${name}`)
    .filter((file) => !seen.has(file))
    .sort()
    .map((file) => ({
      file,
      kind: 'dataset',
      title: titleFromPath(file),
      defaultSelected: DEFAULT_DATASET_SELECTION.has(file),
    }));

  return [...entries, ...extraEntries];
}

function keepExistingDatasetEntries(entries) {
  return entries.filter((entry) => {
    const exists = fs.existsSync(path.join(DOCS_DIR, entry.file));
    if (!exists) {
      console.warn(`Skipping missing dataset file from TOC: ${entry.file}`);
    }
    return exists;
  });
}

function validateEntriesExist(entries) {
  entries.forEach((entry) => {
    if (!fs.existsSync(path.join(DOCS_DIR, entry.file))) {
      throw new Error(`Missing source file: ${entry.file}`);
    }
  });
}

function escapeInlineScript(source) {
  return source.replace(/<\/script/gi, '<\\/script');
}

function escapeInlineStyle(source) {
  return source.replace(/<\/style/gi, '<\\/style');
}

function buildBundleScript(json) {
  return [
    '/* Generated by docs/scripts/build_formalism_download_bundle.mjs. */',
    '(function () {',
    `  window.HYPOSTRUCTURE_FORMALISM_BUNDLE = ${json};`,
    '}());',
    '',
  ].join('\n');
}

function writePromptBuilder(bundleScript) {
  const widgetCss = fs.readFileSync(WIDGET_CSS_FILE, 'utf8');
  const widgetJs = fs.readFileSync(WIDGET_JS_FILE, 'utf8');

  fs.writeFileSync(
    BUILDER_FILE,
    [
      '<!doctype html>',
      '<html lang="en">',
      '<head>',
      '  <meta charset="utf-8">',
      '  <meta name="viewport" content="width=device-width, initial-scale=1">',
      '  <title>Hypostructure Prompt Builder</title>',
      '  <style>',
      '    body {',
      '      background: #f3f4f6;',
      '      color: #111827;',
      '      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;',
      '      margin: 0;',
      '    }',
      '',
      '    main {',
      '      margin: 0 auto;',
      '      max-width: 1200px;',
      '      padding: 2rem 1rem;',
      '    }',
      '',
      '    h1 {',
      '      font-size: clamp(2rem, 5vw, 3.75rem);',
      '      line-height: 1;',
      '      margin: 0;',
      '    }',
      '',
      '    .hs-builder-intro {',
      '      color: #374151;',
      '      font-size: 1.05rem;',
      '      margin: 0.85rem 0 0;',
      '      max-width: 44rem;',
      '    }',
      '',
      escapeInlineStyle(widgetCss),
      '  </style>',
      '</head>',
      '<body>',
      '  <main>',
      '    <h1>Hypostructure Prompt Builder</h1>',
      '    <p class="hs-builder-intro">',
      '      Choose formalism chapters, dataset documents, and proof granularity. The default export downloads as Markdown.',
      '    </p>',
      '    <div id="hypostructure-download-widget"></div>',
      '  </main>',
      '  <script>',
      escapeInlineScript(bundleScript),
      '  </script>',
      '  <script>',
      escapeInlineScript(widgetJs),
      '  </script>',
      '</body>',
      '</html>',
      '',
    ].join('\n'),
  );
}

function main() {
  const configText = fs.readFileSync(MYST_CONFIG, 'utf8');
  const formalismEntries = appendTemplate(withDefaults(parseFormalismToc(configText), 'formalism', true));
  validateEntriesExist(formalismEntries);
  const datasetEntries = appendDatasetDirectoryFiles(
    keepExistingDatasetEntries(parseDatasetToc(configText).map((entry) => ({
      ...entry,
      kind: 'dataset',
      defaultSelected: DEFAULT_DATASET_SELECTION.has(entry.file),
    }))),
  );
  const entries = [...formalismEntries, ...datasetEntries];
  const bundle = entries.map(readEntry);
  const json = JSON.stringify(bundle, null, 2)
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');

  const bundleScript = buildBundleScript(json);

  fs.mkdirSync(path.dirname(OUTPUT_FILE), { recursive: true });
  fs.writeFileSync(OUTPUT_FILE, bundleScript);
  writePromptBuilder(bundleScript);

  console.log(
    `Wrote ${path.relative(DOCS_DIR, OUTPUT_FILE)} and ${path.relative(DOCS_DIR, BUILDER_FILE)} with ` +
    `${formalismEntries.length} formalism chapters and ${datasetEntries.length} dataset documents.`,
  );
}

main();
