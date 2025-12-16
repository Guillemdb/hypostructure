# Makefile for diagram generation

DIAGRAM_MMD = diagram_framework.mmd
DIAGRAM_PDF = diagram_framework.pdf

.PHONY: diagram clean-diagram

# Render mermaid diagram to PDF using mmdc (Mermaid CLI)
diagram: $(DIAGRAM_PDF)

$(DIAGRAM_PDF): $(DIAGRAM_MMD)
	@echo "=== Rendering mermaid diagram to PDF ==="
	mmdc -i $(DIAGRAM_MMD) -o $(DIAGRAM_PDF)
	@echo "=== Done: $(DIAGRAM_PDF) ==="

clean-diagram:
	rm -f $(DIAGRAM_PDF)
