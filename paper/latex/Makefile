# Build: make          (pdflatex + bibtex, 2-pass)
# Clean: make clean
# Watch: make watch    (requires latexmk)

MAIN = main

.PHONY: all clean watch

all:
	pdflatex $(MAIN).tex
	bibtex   $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

watch:
	latexmk -pdf -pvc $(MAIN).tex

clean:
	rm -f $(MAIN).pdf $(MAIN).aux $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).log $(MAIN).out $(MAIN).toc \
	      sections/*.aux
