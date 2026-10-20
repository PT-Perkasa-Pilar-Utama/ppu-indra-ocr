# Indra OCR models v1.1 (mirror)

This folder is an unmodified mirror of the Indra OCR 1.1 model weights by
Datalab.

- Source: https://huggingface.co/datalab-to/indra-ocr
- Revision: commit `a1283c4b42a273c09480ad50e71461daff50fcc7` (tag `v1.1.0`)
- License: the modified AI Pubs Open RAIL-M license in [LICENSE](LICENSE).
  It is the same file as in the v3.1 revision. It is not the license of this
  repository's code.
- Changes: none. `tests/model-mirror.test.ts` checks every file against the
  upstream sha256.

The `.npz` files are n-gram language models for lines the recognizers are
unsure of. The math files are the same as in v3.1.

The use restrictions in [../README.md](../README.md#use-restrictions) apply.
