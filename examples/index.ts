/**
 * Read a page with ppu-indra-ocr: the whole pipeline in one call, then the
 * same page stage by stage.
 *
 *   bun examples/index.ts [image]
 *
 * The first run downloads the models (about 18 MB) and caches them in
 * ~/.cache/ppu-indra-ocr. Subsequent runs read from disk.
 */

import { IndraOcrService, toMarkdown } from "../src";

const file = process.argv[2] ?? `${import.meta.dir}/../tests/fixtures/pages/two-column.png`;
const image = await Bun.file(file).arrayBuffer();

const service = new IndraOcrService();
await service.initialize();

// The whole pipeline: orient, deskew, detect, recognize, order, lay out.
const page = await service.convert(image);
console.log(page.markdown);
console.log(
  `\n${page.lines.length} lines, ${page.blocks.length} blocks, rotation ${page.rotation}, skew ${page.skew}`
);

// The same page, one stage at a time.
const { turns } = await service.orient(image);
const { image: upright } = await service.straighten(image, turns);
const boxes = (await service.detect(upright)).map(([x0, y0, x1, y1]) => [x0, y0, x1, y1] as const);
const lines = await service.recognize(upright, boxes);
const texts = lines.map((l) => l.text);
const { order, cls, same } = await service.order(boxes, texts, upright.width, upright.height);
if (cls && same) console.log(toMarkdown(boxes, texts, order, cls, same).slice(0, 200));

await service.destroy();
