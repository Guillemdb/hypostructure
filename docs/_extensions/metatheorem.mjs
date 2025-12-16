/**
 * MyST plugin for prf:metatheorem directive
 * Renders metatheorems similar to theorems but with distinct styling
 */

const metatheorem = {
  name: 'prf:metatheorem',
  doc: 'A metatheorem directive for higher-level mathematical statements',
  arg: { type: String, doc: 'The title of the metatheorem' },
  options: {
    label: { type: String, doc: 'Label for cross-referencing' },
    class: { type: String, doc: 'Additional CSS classes' },
  },
  body: { type: 'parsed', doc: 'The content of the metatheorem (parsed as MyST)' },
  run(data) {
    const title = data.arg || 'Metatheorem';
    const label = data.options?.label;
    const customClass = data.options?.class || '';

    // Create the admonition-like structure
    const children = [];

    // Add title
    children.push({
      type: 'admonitionTitle',
      children: [{ type: 'text', value: `Metatheorem: ${title}` }],
    });

    // Add parsed body content directly (already parsed as MyST AST)
    if (data.body && Array.isArray(data.body)) {
      children.push(...data.body);
    }

    const node = {
      type: 'admonition',
      kind: 'metatheorem',
      class: `metatheorem ${customClass}`.trim(),
      children,
    };

    // Add label/identifier for cross-referencing
    if (label) {
      node.identifier = label;
      node.label = label;
    }

    return [node];
  },
};

const plugin = {
  name: 'Metatheorem Directive Plugin',
  directives: [metatheorem],
};

export default plugin;
