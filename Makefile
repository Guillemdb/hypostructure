.PHONY: docs serve

DOCS_DIR := docs
DOCS_HTML := $(DOCS_DIR)/_build/html
MYST := ./node_modules/.bin/myst
MYST_BUNDLE := $(DOCS_DIR)/node_modules/mystmd/dist/myst.cjs

$(MYST_BUNDLE): $(DOCS_DIR)/package.json $(DOCS_DIR)/package-lock.json
	cd $(DOCS_DIR) && npm install

docs: $(MYST_BUNDLE)
	python3 -c "import shutil; shutil.rmtree('$(DOCS_HTML)', ignore_errors=True)"
	cd $(DOCS_DIR) && $(MYST) build --html

serve: docs
	cd $(DOCS_HTML) && python3 -m http.server 8000
