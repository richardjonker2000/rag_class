# make          build slides -> out/rag_class.{deckcraft,pptx,pdf} and PNG previews in out/png/
DECKCRAFT_CLI ?= $(shell ./get_deckcraft.sh)
export DECKCRAFT_CLI
PY ?= ../.venv/bin/python

.PHONY: all
all:
	$(PY) build.py out/rag_class
	$(DECKCRAFT_CLI) convert out/rag_class.deckcraft out/rag_class.pptx
	$(DECKCRAFT_CLI) convert out/rag_class.deckcraft out/rag_class.pdf
	rm -rf out/png && $(DECKCRAFT_CLI) render out/rag_class.pptx --all --scale 1 out/png
