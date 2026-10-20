// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/** Any typed array an .npy dtype maps to. */
export type NpyData =
  | Float32Array
  | Float64Array
  | Int32Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array;

/** One array of an .npz: its shape ([] for a scalar) and data in C order. */
export type NpyArray = { shape: number[]; data: NpyData };

const u16 = (v: DataView, o: number): number => v.getUint16(o, true);
const u32 = (v: DataView, o: number): number => v.getUint32(o, true);
// ponytail: sizes and offsets above 2^53 are not supported; no model file is near that.
const u64 = (v: DataView, o: number): number => Number(v.getBigUint64(o, true));

type Entry = { method: number; csize: number; usize: number; offset: number };

/** The zip64 end-of-central-directory fields, when the plain record is saturated. */
function zip64Eocd(v: DataView, eocd: number) {
  const loc = eocd - 20;
  if (loc < 0 || u32(v, loc) !== 0x07064b50) throw new Error("npz: zip64 locator is missing");
  const rec = u64(v, loc + 8);
  if (rec + 56 > v.byteLength || u32(v, rec) !== 0x06064b50) {
    throw new Error("npz: zip64 end record is missing");
  }
  return { count: u64(v, rec + 32), dirOffset: u64(v, rec + 48) };
}

function readDirectory(bytes: Uint8Array): Map<string, Entry> {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 22 - 0xffff); i--) {
    if (u32(v, i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("npz: not a zip archive (no end-of-central-directory record)");
  let count = u16(v, eocd + 10);
  let pos = u32(v, eocd + 16);
  if (count === 0xffff || pos === 0xffffffff) ({ count, dirOffset: pos } = zip64Eocd(v, eocd));

  const dec = new TextDecoder();
  const out = new Map<string, Entry>();
  for (let n = 0; n < count; n++) {
    if (pos + 46 > bytes.length || u32(v, pos) !== 0x02014b50) {
      throw new Error("npz: bad central directory entry");
    }
    const flags = u16(v, pos + 8);
    if (flags & 1) throw new Error("npz: encrypted entries are not supported");
    const method = u16(v, pos + 10);
    let csize = u32(v, pos + 20);
    let usize = u32(v, pos + 24);
    const nameLen = u16(v, pos + 28);
    const extraLen = u16(v, pos + 30);
    const commentLen = u16(v, pos + 32);
    let offset = u32(v, pos + 42);
    const name = dec.decode(bytes.subarray(pos + 46, pos + 46 + nameLen));
    // zip64 extra field: the saturated fields, in this order
    let x = pos + 46 + nameLen;
    const xEnd = x + extraLen;
    while (x + 4 <= xEnd) {
      const id = u16(v, x);
      const len = u16(v, x + 2);
      if (id === 0x0001) {
        let p = x + 4;
        if (usize === 0xffffffff) {
          usize = u64(v, p);
          p += 8;
        }
        if (csize === 0xffffffff) {
          csize = u64(v, p);
          p += 8;
        }
        if (offset === 0xffffffff) offset = u64(v, p);
      }
      x += 4 + len;
    }
    out.set(name, { method, csize, usize, offset });
    pos += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}

async function inflateRaw(data: Uint8Array, usize: number): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("npz: this runtime has no DecompressionStream, which Deflate entries need");
  }
  const ds = new DecompressionStream("deflate-raw");
  const w = ds.writable.getWriter();
  // A failure here also fails the read below, which reports it.
  const feed = async (): Promise<void> => {
    try {
      // SAFETY: a Uint8Array over a plain ArrayBuffer; the DOM type wants that narrower form.
      await w.write(data as Uint8Array<ArrayBuffer>);
      await w.close();
    } catch {}
  };
  void feed();
  const out = new Uint8Array(usize);
  const r = ds.readable.getReader();
  let n = 0;
  for (;;) {
    const { done, value } = await r.read();
    if (done) break;
    if (n + value.length > usize) throw new Error("npz: entry is larger than its header says");
    out.set(value, n);
    n += value.length;
  }
  if (n !== usize) throw new Error("npz: entry is smaller than its header says");
  return out;
}

async function entryBytes(bytes: Uint8Array, name: string, e: Entry): Promise<Uint8Array> {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (e.offset + 30 > bytes.length || u32(v, e.offset) !== 0x04034b50) {
    throw new Error(`npz: bad local header for ${name}`);
  }
  const start = e.offset + 30 + u16(v, e.offset + 26) + u16(v, e.offset + 28);
  if (start + e.csize > bytes.length) throw new Error(`npz: ${name} is truncated`);
  const raw = bytes.subarray(start, start + e.csize);
  if (e.method === 0) return raw;
  if (e.method === 8) return inflateRaw(raw, e.usize);
  throw new Error(`npz: ${name} uses unsupported compression method ${e.method}`);
}

type Dtype = { size: number; make: (b: ArrayBuffer, o: number, n: number) => NpyData };

function dtypeFor(descr: string): Dtype | undefined {
  switch (descr) {
    case "<f4":
      return { size: 4, make: (b, o, n) => new Float32Array(b, o, n) };
    case "<f8":
      return { size: 8, make: (b, o, n) => new Float64Array(b, o, n) };
    case "<i4":
      return { size: 4, make: (b, o, n) => new Int32Array(b, o, n) };
    case "<u4":
      return { size: 4, make: (b, o, n) => new Uint32Array(b, o, n) };
    case "<i8":
      return { size: 8, make: (b, o, n) => new BigInt64Array(b, o, n) };
    case "<u8":
      return { size: 8, make: (b, o, n) => new BigUint64Array(b, o, n) };
    default:
      return undefined;
  }
}

function parseNpy(name: string, buf: Uint8Array): NpyArray {
  const fail = (why: string): never => {
    throw new Error(`npz: ${name}: ${why}`);
  };
  if (
    buf.length < 10 ||
    buf[0] !== 0x93 ||
    String.fromCharCode(...buf.subarray(1, 6)) !== "NUMPY"
  ) {
    fail("not an .npy array");
  }
  const v = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const major = buf[6];
  let hdrLen: number;
  let hdrAt: number;
  if (major === 1) {
    hdrLen = u16(v, 8);
    hdrAt = 10;
  } else if (major === 2 || major === 3) {
    hdrLen = u32(v, 8);
    hdrAt = 12;
  } else {
    return fail(`unsupported .npy version ${major}`);
  }
  if (hdrAt + hdrLen > buf.length) fail("header is truncated");
  const header = new TextDecoder(major === 3 ? "utf-8" : "latin1").decode(
    buf.subarray(hdrAt, hdrAt + hdrLen)
  );
  const descr = /'descr':\s*'([^']*)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1];
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  if (descr === undefined || fortran === undefined || shapeText === undefined) {
    return fail("cannot read the header");
  }
  if (fortran === "True") fail("fortran_order arrays are not supported");
  const dtype = dtypeFor(descr);
  if (!dtype) return fail(`unsupported dtype ${descr}`);
  const shape = shapeText
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "")
    .map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  const dataAt = hdrAt + hdrLen;
  if (dataAt + count * dtype.size > buf.length) fail("data is truncated");
  // Typed arrays need an aligned offset; copy when the entry is not.
  const abs = buf.byteOffset + dataAt;
  // SAFETY: bytes come from our own Uint8Array/inflate output, never a SharedArrayBuffer.
  const whole = buf.buffer as ArrayBuffer;
  const data =
    abs % dtype.size === 0
      ? dtype.make(whole, abs, count)
      : // SAFETY: slice() copies into a fresh ArrayBuffer.
        dtype.make(buf.slice(dataAt, dataAt + count * dtype.size).buffer as ArrayBuffer, 0, count);
  return { shape, data };
}

/**
 * Reads an .npz archive (stored or Deflate entries, zip64 included) into a map
 * from array name (without ".npy") to its shape and typed data. The data is
 * little-endian, as numpy writes it, so it assumes a little-endian runtime.
 * "<u8" and "<i8" arrays come as BigUint64Array and BigInt64Array.
 */
export async function readNpz(bytes: Uint8Array): Promise<Map<string, NpyArray>> {
  const out = new Map<string, NpyArray>();
  for (const [name, entry] of readDirectory(bytes)) {
    if (!name.endsWith(".npy")) continue;
    out.set(name.slice(0, -4), parseNpy(name, await entryBytes(bytes, name, entry)));
  }
  return out;
}
