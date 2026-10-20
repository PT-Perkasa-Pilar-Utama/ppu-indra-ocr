// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/**
 * Usage text for the `ppu-indra-ocr` CLI. Printed to stdout for `help` /
 * `--help`, and to stderr (followed by a non-zero exit) on usage errors.
 */

export const USAGE = `ppu-indra-ocr - Indra OCR on the command line

Usage:
  ppu-indra-ocr <command> [args] [flags]

Commands:
  convert <file...>      OCR images and PDFs (paths, globs, or http(s) URLs)
  detect <image>         Detect text lines only (JSON boxes to stdout)
  download-models        Pre-warm the model cache (~/.cache/ppu-indra-ocr)
  clear-cache            Delete the cached model files
  models                 Print the model presets, their files, URLs and revisions
  help                   Show this help
  version                Show the installed version

Convert flags:
  --format <f>           md | text | json  (default md; json is one object per page)
  --out <dir>            write one file per input (.md, .txt or .jsonl)
  --math                 write math as LaTeX (loads 4 more models)
  --keep-furniture       keep page headers and footers in the markdown
  --engine <e>           opencv | canvas-native  (default opencv)
  --timings              print per-page source and time to stderr

PDF flags:
  --mode <m>             auto | ocr | text  (default auto: text layer where sound, OCR elsewhere)
  --dpi <n>              render resolution for OCR (default 200)
  --max-pages <n>        read at most n pages per PDF
  --password <p>         password for encrypted PDFs
  --no-image-ocr         on text-layer pages, do not OCR text inside images

Models and session:
  --model <preset>       v1.1 | v3.1  (default v1.1; v3.1 is the smaller first release)
  --no-lm                do not re-read unsure lines with the language models
  --model-dir <dir>      read the model files from a folder instead of downloading
  --execution-providers <list>  comma-separated, e.g. cpu or cuda,cpu  (default cpu)

Output:
  -o, --output <file>    write everything to one file instead of stdout
  --pretty               indent JSON output
  -q, --quiet            suppress progress on stderr
  --verbose              log each step to stderr

Examples:
  ppu-indra-ocr convert paper.pdf
  ppu-indra-ocr convert "scans/*.png" --out results/
  ppu-indra-ocr convert scan.png --format json --pretty
  ppu-indra-ocr convert paper.pdf --mode ocr --math --timings
  ppu-indra-ocr detect receipt.jpg

The model weights are Datalab's (https://huggingface.co/datalab-to/indra-ocr),
under a modified OpenRAIL-M license with use restrictions.
`;
