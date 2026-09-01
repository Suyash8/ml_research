#!/usr/bin/env bash
# Build a manuscript version and summarise the LaTeX log.
#
#   scripts/build_paper.sh manuscript/v3-simplified
#
# Runs the full pdflatex/bibtex/pdflatex/pdflatex cycle, keeps intermediate
# files in <dir>/build, and prints only the lines that matter: undefined
# citations, undefined references, and errors.

set -uo pipefail

DIR="${1:-manuscript/v3-simplified}"
JOB="paper"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ABS="$ROOT/$DIR"

if [ ! -f "$ABS/$JOB.tex" ]; then
  echo "no such file: $ABS/$JOB.tex" >&2
  exit 1
fi

mkdir -p "$ABS/build"
cd "$ABS" || exit 1

run() {
  timeout 120 pdflatex -interaction=nonstopmode -file-line-error \
    -output-directory=build "$JOB.tex" </dev/null >/dev/null 2>&1
}

echo "== pass 1 =="; run
echo "== bibtex =="
# bibtex needs the .bib and .aux discoverable from its working directory
BIBINPUTS="$ABS:$ROOT/manuscript/shared:" timeout 60 bibtex "build/$JOB" \
  </dev/null > "build/$JOB.bibtex.out" 2>&1 || true
sed -n '1,40p' "build/$JOB.bibtex.out"
echo "== pass 2 =="; run
echo "== pass 3 =="; run

cp -f "build/$JOB.pdf" "$JOB.pdf" 2>/dev/null && echo "wrote $DIR/$JOB.pdf"

LOG="build/$JOB.log"
count() { grep -c "$1" "$LOG" 2>/dev/null | head -1; }

echo
echo "================ LOG SUMMARY ================"
echo "undefined citations : $(count 'Citation.*undefined')"
grep -o "Citation \`[^']*' undefined" "$LOG" 2>/dev/null | sort -u | sed 's/^/    /'
echo "undefined references: $(count 'Reference.*undefined')"
grep -o "Reference \`[^']*' undefined" "$LOG" 2>/dev/null | sort -u | sed 's/^/    /'
echo "overfull hboxes     : $(count 'Overfull .hbox')"
echo "errors              : $(count '^!')"
grep -n "^!" "$LOG" 2>/dev/null | head -20 | sed 's/^/    /'
echo "bibtex warnings     : $(grep -c 'Warning' "build/$JOB.blg" 2>/dev/null | head -1)"
grep "Warning" "build/$JOB.blg" 2>/dev/null | head -10 | sed 's/^/    /'
echo
grep -o "Output written.*" "$LOG" 2>/dev/null | tail -1
echo "============================================="
