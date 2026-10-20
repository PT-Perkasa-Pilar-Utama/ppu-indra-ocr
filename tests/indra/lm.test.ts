import { describe, expect, test } from "bun:test";
import { readFileSync } from "fs";
import { join } from "path";
import { deflateRawSync } from "zlib";

import { beamSearch, NgramLM } from "../../src/indra/lm.js";
import { readNpz } from "../../src/indra/npz.js";
import { fetchAndCacheResource } from "../../src/processor/model-cache.js";

/**
 * The n-gram model and the CTC prefix beam search against goldens from the
 * Python reference (indra-ocr 1.1.1).
 */

const GOLDEN_DIR = join(import.meta.dir, "..", "fixtures", "golden");
const MODELS_DIR = join(import.meta.dir, "..", "..", "models", "v1.1");
const LFS_POINTER = "version https://git-lfs.github.com/spec/v1";
const LM_URL =
  "https://huggingface.co/datalab-to/indra-ocr/resolve/a1283c4b42a273c09480ad50e71461daff50fcc7/";

/** The mirrored file when it is real, else the cached download (CI has LFS pointers). */
async function modelBytes(file: string): Promise<Uint8Array> {
  const local = readFileSync(join(MODELS_DIR, file));
  if (!local.subarray(0, LFS_POINTER.length).toString().startsWith(LFS_POINTER)) {
    return new Uint8Array(local.buffer, local.byteOffset, local.byteLength);
  }
  return new Uint8Array(await fetchAndCacheResource(LM_URL + file));
}

const models = new Map<string, Promise<NgramLM>>();
function loadLm(file: string): Promise<NgramLM> {
  let p = models.get(file);
  if (!p) {
    p = (async () => NgramLM.fromNpz(await modelBytes(file)))();
    models.set(file, p);
  }
  return p;
}

function readJson<T>(name: string): T {
  // SAFETY: the file is written by scripts/golden in this shape.
  return JSON.parse(readFileSync(join(GOLDEN_DIR, name), "utf8")) as T;
}

function b64(s: string): Buffer {
  return Buffer.from(s, "base64");
}

/** Builds a one-array .npy (version 1). */
function npy(descr: string, shape: string, data: Uint8Array): Uint8Array {
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shape}, }`;
  while ((10 + header.length + 1) % 64) header += " ";
  header += "\n";
  const out = new Uint8Array(10 + header.length + data.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length, 0]);
  out.set(new TextEncoder().encode(header), 10);
  out.set(data, 10 + header.length);
  return out;
}

/** Builds a zip with the given entries, each stored (method 0) or deflated (8). */
function zip(entries: { name: string; data: Uint8Array; deflate: boolean }[]): Uint8Array {
  const parts: Buffer[] = [];
  const dir: Buffer[] = [];
  let offset = 0;
  for (const e of entries) {
    const name = Buffer.from(e.name);
    const body = e.deflate ? deflateRawSync(e.data) : Buffer.from(e.data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(e.deflate ? 8 : 0, 8);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(e.data.length, 22);
    local.writeUInt16LE(name.length, 26);
    const cd = Buffer.alloc(46);
    cd.writeUInt32LE(0x02014b50, 0);
    cd.writeUInt16LE(e.deflate ? 8 : 0, 10);
    cd.writeUInt32LE(body.length, 20);
    cd.writeUInt32LE(e.data.length, 24);
    cd.writeUInt16LE(name.length, 28);
    cd.writeUInt32LE(offset, 42);
    dir.push(cd, name);
    parts.push(local, name, body);
    offset += 30 + name.length + body.length;
  }
  const cdBuf = Buffer.concat(dir);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(cdBuf.length, 12);
  end.writeUInt32LE(offset, 16);
  return new Uint8Array(Buffer.concat([...parts, cdBuf, end]));
}

describe("readNpz", () => {
  test("reads stored and deflated entries, scalars and arrays", async () => {
    const f4 = new Uint8Array(new Float32Array([1.5, -2, 3.25]).buffer);
    const u8 = new Uint8Array(new BigUint64Array([1n, 2n ** 60n + 5n]).buffer);
    const z = await readNpz(
      zip([
        { name: "a.npy", data: npy("<f4", "(3,)", f4), deflate: true },
        { name: "b.npy", data: npy("<u8", "(2,)", u8), deflate: false },
        {
          name: "n.npy",
          data: npy("<i8", "()", new Uint8Array(new BigInt64Array([7n]).buffer)),
          deflate: true,
        },
      ])
    );
    expect([...z.keys()]).toEqual(["a", "b", "n"]);
    expect(z.get("a")?.shape).toEqual([3]);
    expect([...(z.get("a")?.data ?? [])]).toEqual([1.5, -2, 3.25]);
    expect(z.get("b")?.data).toEqual(new BigUint64Array([1n, 2n ** 60n + 5n]));
    expect(z.get("n")?.shape).toEqual([]);
    expect(z.get("n")?.data).toEqual(new BigInt64Array([7n]));
  });

  test("rejects unsupported input", async () => {
    await expect(readNpz(new Uint8Array(10))).rejects.toThrow(/not a zip/);
    const be = zip([
      { name: "x.npy", data: npy(">f4", "(1,)", new Uint8Array(4)), deflate: false },
    ]);
    await expect(readNpz(be)).rejects.toThrow(/unsupported dtype/);
  });

  test("a model file has the shapes numpy reports", async () => {
    const z = await readNpz(await modelBytes("lm.npz"));
    expect(Number(z.get("order")?.data[0])).toBe(5);
    expect(z.get("order")?.shape).toEqual([]);
    expect(z.get("k4")?.shape).toEqual([1756113]);
    expect(z.get("k4")?.data).toBeInstanceOf(BigUint64Array);
    expect(z.get("hb0")?.data).toBeInstanceOf(Float32Array);
  });
});

type LmGolden = {
  file: string;
  order: number;
  bits: number;
  vocab_size: number;
  max_logp: number;
  keys: string[];
  logp: number[];
};

describe("NgramLM", () => {
  for (const name of ["lm", "lm_cjk", "lm_world"]) {
    test(`${name}: header, max_logp and logp match the reference`, async () => {
      const g = readJson<LmGolden>(`lm.${name}.json`);
      const lm = await loadLm(g.file);
      expect(lm.order).toBe(g.order);
      expect(lm.bits).toBe(g.bits);
      expect(lm.vocabSize).toBe(g.vocab_size);
      expect(Math.abs(lm.maxLogp - g.max_logp)).toBeLessThan(1e-9);
      const got = lm.logpKeys(g.keys.map((k) => BigInt(k)));
      expect(got.length).toBe(g.logp.length);
      let worst = 0;
      got.forEach((v, i) => {
        worst = Math.max(worst, Math.abs(v - (g.logp[i] as number)));
      });
      expect(worst).toBeLessThan(1e-9);
    });
  }
});

type BeamGolden = {
  case: string;
  lm: string;
  T: number;
  k: number;
  alpha: number;
  beta: number;
  beam: number;
  cand_floor: number;
  ids: number[];
  steps: number[];
  blank: string;
  topv: string;
  topi: string;
};

describe("beamSearch", () => {
  const calls = readJson<BeamGolden[]>("beam-search.json");
  calls.forEach((c, n) => {
    test(`${c.case} #${n} (${c.lm}, T=${c.T}) matches the reference`, async () => {
      const lm = await loadLm(c.lm);
      const blank = new Float32Array(new Uint8Array(b64(c.blank)).buffer);
      const topv = new Float32Array(new Uint8Array(b64(c.topv)).buffer);
      const topi = new BigInt64Array(new Uint8Array(b64(c.topi)).buffer);
      const r = beamSearch(
        blank,
        topv,
        topi,
        c.T,
        c.k,
        lm,
        c.alpha,
        c.beta,
        c.beam,
        0.999,
        c.cand_floor
      );
      expect(r.ids).toEqual(c.ids);
      expect(r.steps).toEqual(c.steps);
    });
  });
});
