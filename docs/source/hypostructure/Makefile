# Makefile for Hypostructures PDF compilation
# Uses pandoc + lualatex + biber for bibliography support

# Source files
MD_SOURCE = final_hypostructure.md
BIB_SOURCE = references.bib
TEX_INTERMEDIATE = hypostructures.tex
PDF_OUTPUT = hypostructures.pdf

# Pandoc options
PANDOC_OPTS = --standalone \
              --from markdown+tex_math_single_backslash \
              --top-level-division=part \
              --number-sections \
              --toc \
              --biblatex \
              --bibliography=$(BIB_SOURCE)

# LaTeX options
LUALATEX_OPTS = -interaction=nonstopmode -halt-on-error

.PHONY: pdf clean tex

# Default target
pdf: $(PDF_OUTPUT)

# Full compilation with bibliography
$(PDF_OUTPUT): $(TEX_INTERMEDIATE) $(BIB_SOURCE)
	@echo "=== Running lualatex (pass 1) ==="
	lualatex $(LUALATEX_OPTS) $(TEX_INTERMEDIATE)
	@echo "=== Running biber ==="
	biber hypostructures
	@echo "=== Running lualatex (pass 2) ==="
	lualatex $(LUALATEX_OPTS) $(TEX_INTERMEDIATE)
	@echo "=== Running lualatex (pass 3 - final) ==="
	lualatex $(LUALATEX_OPTS) $(TEX_INTERMEDIATE)
	@echo "=== Done: $(PDF_OUTPUT) ==="

# Generate LaTeX from Markdown
$(TEX_INTERMEDIATE): $(MD_SOURCE)
	@echo "=== Converting Markdown to LaTeX ==="
	pandoc $(MD_SOURCE) -o $(TEX_INTERMEDIATE) $(PANDOC_OPTS)

# Generate only LaTeX (for debugging)
tex: $(TEX_INTERMEDIATE)

# Clean intermediate files
clean:
	rm -f hypostructures.aux hypostructures.bbl hypostructures.bcf
	rm -f hypostructures.blg hypostructures.log hypostructures.out
	rm -f hypostructures.run.xml hypostructures.toc
	rm -f $(TEX_INTERMEDIATE)

# Clean everything including PDF
cleanall: clean
	rm -f $(PDF_OUTPUT)

# Quick rebuild (skip biber if .bbl exists)
quick:
	lualatex $(LUALATEX_OPTS) $(TEX_INTERMEDIATE)
