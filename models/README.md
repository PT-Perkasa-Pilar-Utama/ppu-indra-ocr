# Indra OCR models (mirror)

This folder is an unmodified mirror of the Indra OCR model weights by
Datalab.

- Source: https://huggingface.co/datalab-to/indra-ocr
- Revisions:
  - This folder: commit `d7c839c892bfe4b8993373db64c62f154d05d70b` (tag
    `v3.1`), the `v3.1` preset (Indra OCR 1.0).
  - [v1.1/](v1.1/): commit `a1283c4b42a273c09480ad50e71461daff50fcc7` (tag
    `v1.1.0`), the `v1.1` preset (Indra OCR 1.1).
  - The `v1.2` preset (Indra OCR 1.2, the default), commit
    `a04fa100e778192aa54d040a983983c6d3ee4ce5` (tag `v1.2.0`), is not
    mirrored here: this repository's git LFS budget cannot hold it. The
    library downloads it from Datalab's repository. `scripts/golden` reads a
    local, git-ignored copy in `v1.2/`.
- License: the modified AI Pubs Open RAIL-M license in [LICENSE](LICENSE).
  It is not the license of this repository's code. The v3.1 and v1.1
  revisions share this file; v1.2 has its own, with lower limits (below).
- Changes: none. `tests/model-mirror.test.ts` checks every file against the
  upstream sha256.

Datalab created and owns these weights. This project is a community project.
Datalab does not endorse it. See [LICENSE-STATUS.md](../LICENSE-STATUS.md) for
our open request to Datalab and what may change.

## Use restrictions

The weights are subject to paragraph 5 and Attachment A of the license. By
using the weights, you agree to these terms. You must also bind your own
users to them. Attachment A, quoted from the license:

> As conditions to the Licenses set forth in this Agreement, You agree not to
> use, reproduce, modify, create or Distribute the Model, Derivatives of the
> Model, or Output (collectively, "Use") in any of the following ways:
>
> 1. Legal:
>    (a) In any way that violates any applicable national, federal, state,
>    local or international law or regulation; or
>    (b) to directly or indirectly infringe or misappropriate any third party
>    intellectual property rights (including those of Licensor or any
>    Contributor)
> 2. Commercial:
>    (a) for any purpose if You (your employer, or the entity you are
>    affiliated with) generated more than twenty million US Dollars
>    ($20,000,000) in gross revenue in the prior year, except where Your Use
>    is limited to personal use or research purposes;
>    (b) for any purpose if You (your employer, or the entity you are
>    affiliated with) has raised more than twenty million US dollars
>    ($20,000,000) in total equity or debt funding from any source, except
>    where Your Use is limited to personal use or research purposes; or
>    (c) for any purpose if You (your employer, or the entity you are
>    affiliated with) provides or otherwise makes available any product or
>    service that competes with any product or service offered by or made
>    available by Licensor or any of its affiliates.
>
> Commercial and broader use licenses may be available from Licensor at the
> following URL: https://www.datalab.to/

The license of the Indra OCR 1.2 weights ([LICENSE at v1.2.0](https://huggingface.co/datalab-to/indra-ocr/blob/a04fa100e778192aa54d040a983983c6d3ee4ce5/LICENSE)) sets
both limits in 2(a) and 2(b) at five million US dollars ($5,000,000); the rest
of Attachment A is the same.

Section 8 of the license applies the same license to the Output of the model.
