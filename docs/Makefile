.PHONY: docs docs-fast serve pdf clean install

NPM_REAL := $(shell command -v npm)
MYST_NODE_OPTIONS := --require=$(CURDIR)/scripts/myst-node-shims.cjs
MYST_ENV := PATH=$(CURDIR)/scripts/myst-bin:$$PATH NPM_REAL=$(NPM_REAL) NODE_OPTIONS=$(MYST_NODE_OPTIONS)

# Build the Jupyter Book
docs:
	npm run build:formalism-bundle
	python3 -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('_build/html', '_build/site')]"
	. .venv/bin/activate && $(MYST_ENV) MYST_LANG=PYTHON node node_modules/mystmd/dist/myst.cjs build

# Build the reduced development docs
docs-fast:
	npm run build:formalism-bundle
	npm run build:serve-config
	python3 -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('_build/html', '_build/site')]"
	. .venv/bin/activate && $(MYST_ENV) MYST_LANG=PYTHON node node_modules/mystmd/dist/myst.cjs --config myst-serve.yml build

# Serve the built site on localhost:8000
serve:
	npm run build:formalism-bundle
	npm run build:serve-config
	. .venv/bin/activate && $(MYST_ENV) MYST_LANG=PYTHON node node_modules/mystmd/dist/myst.cjs --config myst-serve.yml start --port 8000

# Build PDF output
pdf:
	npm run build:formalism-bundle
	. .venv/bin/activate && $(MYST_ENV) MYST_LANG=PYTHON node node_modules/mystmd/dist/myst.cjs build --pdf

# Clean build artifacts
clean:
	rm -rf _build

# Install dependencies
install:
	uv venv .venv
	. .venv/bin/activate && uv pip install jupyter-book sphinxcontrib-bibtex sphinx-design sphinxcontrib-mermaid
