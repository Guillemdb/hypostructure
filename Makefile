.PHONY: docs docs-fast serve code

DOCS_DIR := docs
DOCS_HTML := $(DOCS_DIR)/_build/html
MYST := ./node_modules/.bin/myst
MYST_BUNDLE := $(DOCS_DIR)/node_modules/mystmd/dist/myst.cjs
NPM_REAL := $(shell command -v npm)
MYST_NODE_OPTIONS := --require=$(CURDIR)/$(DOCS_DIR)/scripts/myst-node-shims.cjs
MYST_ENV := PATH=$(CURDIR)/$(DOCS_DIR)/scripts/myst-bin:$$PATH NPM_REAL=$(NPM_REAL) NODE_OPTIONS=$(MYST_NODE_OPTIONS)
CODE_OUT ?= lean/hypostructure_burgers1d_code.md

$(MYST_BUNDLE): $(DOCS_DIR)/package.json $(DOCS_DIR)/package-lock.json
	cd $(DOCS_DIR) && npm install

docs: $(MYST_BUNDLE)
	cd $(DOCS_DIR) && npm run build:formalism-bundle
	python3 -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('$(DOCS_DIR)/_build/html', '$(DOCS_DIR)/_build/site')]"
	cd $(DOCS_DIR) && $(MYST_ENV) $(MYST) build --html

docs-fast: $(MYST_BUNDLE)
	cd $(DOCS_DIR) && npm run build:formalism-bundle
	cd $(DOCS_DIR) && npm run build:serve-config
	python3 -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('$(DOCS_DIR)/_build/html', '$(DOCS_DIR)/_build/site')]"
	cd $(DOCS_DIR) && $(MYST_ENV) $(MYST) --config myst-serve.yml build --html

serve: $(MYST_BUNDLE)
	cd $(DOCS_DIR) && npm run build:formalism-bundle
	cd $(DOCS_DIR) && npm run build:serve-config
	cd $(DOCS_DIR) && $(MYST_ENV) $(MYST) --config myst-serve.yml start --port 8000

code:
	scripts/build_lean_code_bundle.sh "$(CODE_OUT)"
