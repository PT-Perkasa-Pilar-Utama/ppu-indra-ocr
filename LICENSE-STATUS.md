# License status

> **Subject to change.** This page records where this project stands on the
> license of the Indra OCR model weights. It changes when Datalab answers us
> or when Datalab changes its license. Last updated: 2026-10-17.
>
> This page is not legal advice. Read the license yourself before you use the
> weights.

## Two licenses apply

| Part                                  | Owner                  | License                                                            |
| ------------------------------------- | ---------------------- | ------------------------------------------------------------------ |
| The code in this repository           | PT Perkasa Pilar Utama | Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE)            |
| The Indra OCR model weights (`.onnx`) | Datalab                | Modified AI Pubs Open RAIL-M, see [models/LICENSE](models/LICENSE) |

The Apache-2.0 license of the code does not cover the weights. The library
cannot run without the weights, so the weights license applies to every user
who runs it.

## What the weights license asks of you

When you download or run the weights, you are the licensee. The library
downloads them for you on first use. The obligations are yours, not ours.

Attachment A of the license forbids any use of the weights in these cases:

- **Revenue limit.** You, your employer, or the entity you are affiliated
  with had more than USD 20,000,000 in gross revenue in the prior year.
  Personal use and research are excepted.
- **Funding limit.** That entity has raised more than USD 20,000,000 in
  total equity or debt funding. Personal use and research are excepted.
- **Competitors.** That entity provides or otherwise makes available any
  product or service that competes with any product or service of Datalab or
  its affiliates. This clause has **no** exception for personal use or
  research. The license does not define "competes". Datalab sells document
  OCR and data extraction.

Other terms to read:

- **Section 4.** If you redistribute the weights, or run a service on them
  for others, you must pass the use restrictions on to your users as an
  enforceable term and give them a copy of the license.
- **Section 8.** The license applies itself to copies, derivatives, and the
  Output of the model. Section 6 says Datalab claims no rights in the
  Output. We do not know how Datalab reads the two together.
- **Section 9.** Datalab can restrict use of the model and update it.
- **Section 10.** The license gives no right to use Datalab's names or
  trademarks.

The full Attachment A is quoted in [models/README.md](models/README.md). For
commercial use beyond these terms, contact Datalab at
<https://www.datalab.to/>.

## Our request to Datalab

On 2026-10-14 we emailed Datalab before we started this port. We asked for a
written answer on four points:

1. Does Datalab consider our company a competitor under Attachment A, 2(c)?
   PT Perkasa Pilar Utama publishes free OCR libraries, including
   [ppu-paddle-ocr](https://github.com/PT-Perkasa-Pilar-Utama/ppu-paddle-ocr)
   and its self-hosted REST server. May we publish an Apache-2.0 inference
   library for Indra?
2. May we mirror the unmodified v3.1 weights on our own Hugging Face and
   GitHub repositories, with the license file and credit to Datalab? We
   offered to download them from Datalab's repository instead.
3. Is the name "ppu-indra-ocr" acceptable under Section 10, with a clear
   statement that Datalab does not endorse the project?
4. Does Section 8 apply to the OCR text that users get from the model?

As of 2026-10-17, we have no reply. We do not read silence as permission.
Datalab has not approved or endorsed this project.

## What this project does with the weights

- **Default download.** The library downloads the weights from Datalab's own
  repository,
  [datalab-to/indra-ocr](https://huggingface.co/datalab-to/indra-ocr),
  pinned to commit `d7c839c892bfe4b8993373db64c62f154d05d70b` (tag v3.1).
- **No weights in the package.** The npm and JSR packages contain the code
  only.
- **Mirror.** [models/](models/) in this repository holds an unmodified copy
  of the pinned weights with Datalab's license. The library uses it only
  when you set `PPU_INDRA_OCR_MODEL_MIRROR=1` and Hugging Face fails.
- **REST server image.** The Dockerfiles in [apps/serve](apps/serve) download
  the weights at build time, so a built image contains them. Publishing that
  image, or running it as a service for others, is redistribution under
  Section 4.
- **Name.** "Indra" in the package name refers to the model it runs. This is
  a community project. It is not affiliated with Datalab.

## What can change

If Datalab answers, objects, or changes its license, we will follow its
answer. Possible changes:

- remove the mirror in `models/` and its fallback;
- rename the package;
- change how the library gets the weights;
- archive the project.

We record each change on this page and in [CHANGELOG.md](CHANGELOG.md).

## Contact

Datalab, or anyone with a question about this page, can open an issue on
this repository. We answer license requests from Datalab first.
