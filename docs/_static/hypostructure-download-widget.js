(function () {
  const widgetId = 'hypostructure-download-widget';
  const bundleGlobal = 'HYPOSTRUCTURE_FORMALISM_BUNDLE';

  function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function stripFrontmatter(markdown) {
    return markdown.replace(/^---\n[\s\S]*?\n---\s*/, '');
  }

  function removeDirectiveBlocks(markdown, shouldRemove) {
    const lines = markdown.split('\n');
    const output = [];

    for (let index = 0; index < lines.length;) {
      const line = lines[index];
      const match = line.match(/^(\s*)(:{3,})\{([^}]+)\}/);

      if (match && shouldRemove(lines, index, match)) {
        const closePattern = new RegExp(`^${escapeRegex(match[1])}${escapeRegex(match[2])}\\s*$`);
        index += 1;
        while (index < lines.length && !closePattern.test(lines[index])) index += 1;
        if (index < lines.length) index += 1;
        continue;
      }

      output.push(line);
      index += 1;
    }

    return output.join('\n').replace(/\n{3,}/g, '\n\n');
  }

  function isFeynmanBlock(lines, index) {
    const header = lines.slice(index, Math.min(index + 12, lines.length)).join('\n');
    return /:class:\s+[^\n]*\bfeynman-prose\b/.test(header);
  }

  function isProofBlock(_lines, _index, match) {
    const directiveName = match[3].trim().split(/\s+/)[0];
    return directiveName === 'prf:proof' || directiveName === 'proof';
  }

  function filterMarkdown(markdown, options) {
    let filtered = stripFrontmatter(markdown);

    if (!options.includeFeynman) {
      filtered = removeDirectiveBlocks(filtered, isFeynmanBlock);
    }

    if (!options.includeProofs) {
      filtered = removeDirectiveBlocks(filtered, isProofBlock);
    }

    return filtered.trim();
  }

  function markdownToText(markdown) {
    return markdown
      .replace(/^---\n[\s\S]*?\n---\s*/g, '')
      .replace(/^\s*:{3,}\{prf:([^}\s]+)\}\s*(.*)$/gm, function (_match, kind, title) {
        const label = kind.charAt(0).toUpperCase() + kind.slice(1);
        return title ? `${label}: ${title}` : label;
      })
      .replace(/^\s*:{3,}\{(admonition|dropdown)\}\s*(.*)$/gm, '$2')
      .replace(/^\s*:{3,}\{div\}\s*$/gm, '')
      .split('\n')
      .filter((line) => !/^\s*:{3,}\{/.test(line))
      .filter((line) => !/^\s*:{3,}\s*$/.test(line))
      .filter((line) => !/^\s*:[\w-]+:/.test(line))
      .filter((line) => !/^\s*```/.test(line))
      .join('\n')
      .replace(/^\([^)]+\)=\s*$/gm, '')
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/\{[^}]+\}`([^`]+)`/g, '$1')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/^#{1,6}\s*/gm, '')
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/__([^_]+)__/g, '$1')
      .replace(/\*([^*\n]+)\*/g, '$1')
      .replace(/_([^_\n]+)_/g, '$1')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  function selectedEntries(widget, bundle, kind) {
    const selector = kind ?
      `[data-entry-kind="${kind}"][data-entry-index]:checked` :
      '[data-entry-index]:checked';

    return Array.from(widget.querySelectorAll(selector))
      .map((input) => bundle[Number(input.dataset.entryIndex)])
      .filter(Boolean);
  }

  function currentOptions(widget) {
    const formatInput = widget.querySelector('input[name="hypostructure-export-format"]:checked');
    return {
      includeFeynman: widget.querySelector('[data-option="feynman"]').checked,
      includeProofs: widget.querySelector('[data-option="proofs"]').checked,
      format: formatInput ? formatInput.value : 'markdown',
    };
  }

  function buildExport(entries, options) {
    const optionLabels = [
      options.includeFeynman ? 'Feynman prose included' : 'Feynman prose excluded',
      options.includeProofs ? 'Proof blocks included' : 'Proof blocks excluded',
    ];
    const formalismCount = entries.filter((entry) => entry.kind === 'formalism').length;
    const datasetCount = entries.filter((entry) => entry.kind === 'dataset').length;
    const parts = [
      '# Hypostructure Download Prompt',
      '',
      `Generated: ${new Date().toISOString()}`,
      `Granularity: ${optionLabels.join('; ')}`,
      `Formalism chapters: ${formalismCount}`,
      `Dataset documents: ${datasetCount}`,
      '',
    ];

    entries.forEach((entry, index) => {
      const body = filterMarkdown(entry.content, options);
      const kindLabel = entry.kind === 'dataset' ? 'Dataset' : 'Formalism';
      parts.push('---', '', `# ${index + 1}. [${kindLabel}] ${entry.title}`, '', `_Source: ${entry.path}_`, '', body, '');
    });

    const markdown = parts.join('\n').replace(/\n{4,}/g, '\n\n\n').trim() + '\n';
    return options.format === 'text' ? markdownToText(markdown) + '\n' : markdown;
  }

  function downloadFile(content, format) {
    const extension = format === 'text' ? 'txt' : 'md';
    const mime = format === 'text' ? 'text/plain;charset=utf-8' : 'text/markdown;charset=utf-8';
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');

    link.href = url;
    link.download = `hypostructure-prompt.${extension}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function setMessage(widget, message, tone) {
    const messageEl = widget.querySelector('[data-export-message]');
    messageEl.textContent = message;
    if (tone) messageEl.dataset.tone = tone;
    else delete messageEl.dataset.tone;
  }

  function refreshStatus(widget, bundle) {
    const formalismCount = selectedEntries(widget, bundle, 'formalism').length;
    const datasetCount = selectedEntries(widget, bundle, 'dataset').length;
    const totalCount = formalismCount + datasetCount;
    const formalismTotal = bundle.filter((entry) => entry.kind === 'formalism').length;
    const datasetTotal = bundle.filter((entry) => entry.kind === 'dataset').length;
    const status = widget.querySelector('[data-export-status]');
    const button = widget.querySelector('[data-action="download"]');
    const options = currentOptions(widget);

    status.textContent = `${formalismCount}/${formalismTotal} formalism, ${datasetCount}/${datasetTotal} dataset selected`;
    button.textContent = options.format === 'text' ? 'Download text file' : 'Download Markdown prompt';
    button.disabled = totalCount === 0;
  }

  function renderEntries(widget, bundle, kind) {
    const list = widget.querySelector(`[data-document-list="${kind}"]`);
    list.textContent = '';

    bundle.forEach((entry, index) => {
      if (entry.kind !== kind) return;

      const label = document.createElement('label');
      const checkbox = document.createElement('input');
      const text = document.createElement('span');
      const title = document.createElement('span');
      const sourcePath = document.createElement('span');

      label.className = 'hs-document-row';
      checkbox.type = 'checkbox';
      checkbox.checked = entry.defaultSelected !== false;
      checkbox.value = String(index);
      checkbox.dataset.entryIndex = String(index);
      checkbox.dataset.entryKind = entry.kind;
      title.className = 'hs-document-title';
      title.textContent = entry.title;
      sourcePath.className = 'hs-document-path';
      sourcePath.textContent = entry.path;

      text.append(title, sourcePath);
      label.append(checkbox, text);
      list.appendChild(label);
    });
  }

  function renderShell(widget) {
    widget.className = 'hs-download-widget';
    widget.setAttribute('aria-label', 'Hypostructure prompt download builder');
    widget.innerHTML = [
      '<div class="hs-download-header">',
      '<div><p class="hs-download-kicker">Prompt export</p><h2>Build a single Markdown prompt</h2></div>',
      '<p class="hs-download-status" data-export-status>Loading documents...</p>',
      '</div>',
      '<div class="hs-download-grid">',
      '<section class="hs-download-section" aria-labelledby="hs-formalism-title">',
      '<div class="hs-section-heading"><h3 id="hs-formalism-title">Formalism Chapters</h3>',
      '<div class="hs-button-row" aria-label="Formalism chapter selection controls">',
      '<button type="button" class="hs-button" data-action="select-all" data-target-kind="formalism">All</button>',
      '<button type="button" class="hs-button" data-action="clear" data-target-kind="formalism">Clear</button>',
      '<button type="button" class="hs-button" data-action="restore-default" data-target-kind="formalism">Default</button>',
      '</div></div><div class="hs-document-list" data-document-list="formalism"></div></section>',
      '<section class="hs-download-section" aria-labelledby="hs-dataset-title">',
      '<div class="hs-section-heading"><h3 id="hs-dataset-title">Dataset Documents</h3>',
      '<div class="hs-button-row" aria-label="Dataset document selection controls">',
      '<button type="button" class="hs-button" data-action="select-all" data-target-kind="dataset">All</button>',
      '<button type="button" class="hs-button" data-action="clear" data-target-kind="dataset">Clear</button>',
      '<button type="button" class="hs-button" data-action="restore-default" data-target-kind="dataset">Default</button>',
      '</div></div><div class="hs-document-list" data-document-list="dataset"></div></section>',
      '<section class="hs-download-section" aria-labelledby="hs-options-title">',
      '<h3 id="hs-options-title">Granularity</h3>',
      '<label class="hs-option"><input type="checkbox" data-option="feynman"><span>Include Feynman prose</span></label>',
      '<label class="hs-option"><input type="checkbox" data-option="proofs"><span>Include proof blocks</span></label>',
      '<fieldset class="hs-fieldset"><legend>File type</legend>',
      '<label class="hs-option"><input type="radio" name="hypostructure-export-format" value="markdown" checked><span>Markdown</span></label>',
      '<label class="hs-option"><input type="radio" name="hypostructure-export-format" value="text"><span>Text</span></label>',
      '</fieldset>',
      '<button type="button" class="hs-primary-button" data-action="download">Download Markdown prompt</button>',
      '<p class="hs-download-message" data-export-message role="status" aria-live="polite"></p>',
      '</section></div>',
    ].join('');
  }

  function loadBundle() {
    if (Array.isArray(window[bundleGlobal])) return Promise.resolve(window[bundleGlobal]);

    const bundleLink = document.getElementById('hypostructure-bundle-link') ||
      document.querySelector('a[href*="hypostructure-formalism-bundle"]');
    const source = bundleLink ? bundleLink.href : '_static/hypostructure-formalism-bundle.js';

    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = source;
      script.async = true;
      script.onload = () => {
        if (Array.isArray(window[bundleGlobal])) resolve(window[bundleGlobal]);
        else reject(new Error('The prompt bundle loaded without data.'));
      };
      script.onerror = () => reject(new Error('Could not load the prompt bundle.'));
      document.head.appendChild(script);
    });
  }

  function initWidget() {
    const widget = document.getElementById(widgetId);
    if (!widget || widget.dataset.ready === 'true') return;
    widget.dataset.ready = 'true';
    renderShell(widget);

    loadBundle()
      .then((bundle) => {
        renderEntries(widget, bundle, 'formalism');
        renderEntries(widget, bundle, 'dataset');
        refreshStatus(widget, bundle);

        widget.addEventListener('change', () => {
          refreshStatus(widget, bundle);
          setMessage(widget, '');
        });

        widget.addEventListener('click', (event) => {
          const action = event.target && event.target.dataset ? event.target.dataset.action : undefined;
          if (!action) return;

          const targetKind = event.target.dataset.targetKind;
          const checkboxSelector = targetKind ?
            `[data-entry-kind="${targetKind}"][data-entry-index]` :
            '[data-entry-index]';
          const checkboxes = Array.from(widget.querySelectorAll(checkboxSelector));

          if (action === 'select-all') {
            checkboxes.forEach((checkbox) => { checkbox.checked = true; });
            setMessage(widget, '');
          }

          if (action === 'clear') {
            checkboxes.forEach((checkbox) => { checkbox.checked = false; });
            setMessage(widget, '');
          }

          if (action === 'restore-default') {
            checkboxes.forEach((checkbox) => {
              const entry = bundle[Number(checkbox.dataset.entryIndex)];
              checkbox.checked = entry.defaultSelected !== false;
            });
            setMessage(widget, '');
          }

          if (action === 'download') {
            const entries = selectedEntries(widget, bundle);
            const options = currentOptions(widget);

            if (entries.length === 0) {
              setMessage(widget, 'Select at least one document before downloading.', 'error');
              return;
            }

            downloadFile(buildExport(entries, options), options.format);
            setMessage(widget, `Prepared ${entries.length} document${entries.length === 1 ? '' : 's'}.`);
          }

          refreshStatus(widget, bundle);
        });
      })
      .catch((error) => {
        widget.querySelector('[data-export-status]').textContent = 'Bundle unavailable';
        setMessage(widget, `${error.message} Run npm run build:formalism-bundle from docs/.`, 'error');
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initWidget);
  } else {
    initWidget();
  }
}());
