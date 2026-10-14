// Time convert() per engine on the fixture pages. Small and single-process:
//   ROUNDS=9 bun bench/index.bench.ts
import { IndraOcrService } from "../src";
import { Bench, printResults } from "./harness";

const ROUNDS = Number(process.env.ROUNDS ?? 5);
const page = (name: string) =>
  Bun.file(`${import.meta.dir}/../tests/fixtures/pages/${name}`).arrayBuffer();
const images = {
  "two-column": await page("two-column.png"),
  table: await page("table.png"),
  receipt: await Bun.file(`${import.meta.dir}/../assets/receipt.jpg`).arrayBuffer(),
};

console.log(`\n=== convert() per engine, ${ROUNDS} rounds ===`);
console.log(`${process.platform}/${process.arch}, bun ${Bun.version}\n`);

for (const engine of ["opencv", "canvas-native"] as const) {
  const service = new IndraOcrService({ engine });
  await service.initialize();
  const bench = new Bench({ rounds: ROUNDS, warmup: 1 });
  for (const [name, image] of Object.entries(images)) {
    bench.add(`${engine} ${name}`, () => service.convert(image));
  }
  printResults(await bench.run());
  await service.destroy();
}
