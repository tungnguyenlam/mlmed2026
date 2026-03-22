# Makefile for compiling LaTeX report in practice3

MAIN = main
PDF = $(MAIN).pdf
TEX = $(MAIN).tex
LATEXMK = latexmk -pdf -shell-escape

.PHONY: all clean distclean

all: $(PDF)

$(PDF): $(TEX)
	$(LATEXMK) $(TEX)

clean:
	$(LATEXMK) -c
	rm -f *.aux *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz *.auxlock

distclean: clean
	$(LATEXMK) -C
	rm -f $(PDF)

