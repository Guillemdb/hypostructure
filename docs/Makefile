# Makefile for diagram generation

DIAGRAM_MD = diagram_framework.md
DIAGRAM_PDF = diagram_framework.pdf

.PHONY: diagram clean-diagram

# Render mermaid diagram to PDF
diagram: $(DIAGRAM_PDF)

$(DIAGRAM_PDF): $(DIAGRAM_MD) render_mermaid.py
	@echo "=== Rendering mermaid diagram to PDF ==="
	python3 render_mermaid.py $(DIAGRAM_MD) $(DIAGRAM_PDF)
	@echo "=== Done: $(DIAGRAM_PDF) ==="

clean-diagram:
	rm -f $(DIAGRAM_PDF)
