# ppu-indra-ocr

TypeScript SDK for [Indra OCR](https://github.com/datalab-to/indra) by
Datalab. It runs on Node.js, Bun, Deno, browsers, web workers and React
Native through ONNX Runtime.

> Status: early development. Not released. The API is not stable.

Indra OCR turns images and PDFs into markdown, plain text, or JSON with
boxes. The pipeline:

```
orient -> skew fix -> detect lines -> script check -> recognize (Latin or CJK)
  -> reading order + layout blocks -> markdown
  [optional] math lines -> LaTeX
  [PDF] text layer where it can be trusted, OCR elsewhere
```

## License

- Code: Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
- Model weights: Datalab's modified OpenRAIL-M license, see
  [models/LICENSE](models/LICENSE). It has use restrictions, including
  revenue and funding limits and a ban on use by competitors of Datalab.
  Read [models/README.md](models/README.md) before you use the weights.

This is a community project. Datalab does not endorse it.
