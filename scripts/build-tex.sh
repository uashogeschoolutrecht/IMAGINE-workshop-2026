#!/bin/sh

set -eu

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 presentation/<document>.tex" >&2
  exit 2
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(dirname -- "$script_dir")
presentation_dir="$project_dir/presentation"
source_path=$1

case "$source_path" in
  /*) ;;
  *) source_path="$project_dir/$source_path" ;;
esac

source_dir=$(CDPATH='' cd -- "$(dirname -- "$source_path")" && pwd)
source_name=$(basename -- "$source_path")

if [ "$source_dir" != "$presentation_dir" ]; then
  echo "Refusing to build a TeX file outside $presentation_dir" >&2
  exit 2
fi

case "$source_name" in
  *.tex) document_name=${source_name%.tex} ;;
  *)
    echo "Expected a .tex file, got: $source_name" >&2
    exit 2
    ;;
esac

build_dir=$(mktemp -d "${TMPDIR:-/tmp}/workshop-latex.XXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT HUP INT TERM

(
  cd "$source_dir"
  latexmk \
    -pdf \
    -interaction=nonstopmode \
    -halt-on-error \
    -file-line-error \
    -outdir="$build_dir" \
    "$source_name"
)

mkdir -p "$presentation_dir/pdfs"
install -m 0644 "$build_dir/$document_name.pdf" \
  "$presentation_dir/pdfs/$document_name.pdf"

echo "Created presentation/pdfs/$document_name.pdf"
