.PHONY: docs serve pdf clean install

# Build the Jupyter Book
docs:
	. .venv/bin/activate && jupyter-book build

# Serve the built site on localhost:8000
serve:
	. .venv/bin/activate && jupyter-book start --port 8000

# Build PDF output
pdf:
	. .venv/bin/activate && jupyter-book build --pdf

# Clean build artifacts
clean:
	rm -rf _build

# Install dependencies
install:
	uv venv .venv
	. .venv/bin/activate && uv pip install jupyter-book sphinxcontrib-bibtex sphinx-design sphinxcontrib-mermaid
