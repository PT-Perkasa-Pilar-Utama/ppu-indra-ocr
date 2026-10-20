import { describe, expect, test } from "bun:test";
import { createHash } from "crypto";
import { readdirSync, readFileSync } from "fs";
import { join } from "path";

/**
 * models/ mirrors datalab-to/indra-ocr at d7c839c (tag v3.1) and models/v1.1/
 * at a1283c4 (tag v1.1.0), unmodified.
 * The OpenRAIL-M license requires marking changed files; this test proves
 * there are none. The hashes are the LFS oids from the Hugging Face tree
 * API at that commit (sha256 of the file bytes).
 *
 * CI checks out LFS pointers, not the binaries, to save LFS bandwidth. A
 * pointer already carries the sha256 of its file, so both forms are checked.
 */

const MODELS_DIR = join(import.meta.dir, "..", "models");

const UPSTREAM_SHA256 = {
  LICENSE: "50c88a018a7ee322e89f6b09401b237beb0cd331685879405b3db48993950a5f",
  "det.onnx": "46b0050ce4c4692f887a1a9c038480223af1c7a91b3fcced2657d9cf00290dc4",
  "math_dec_init.onnx": "919417fde9962c3318aeb478c1fe3955dd2bb8cd34476c68debed2ba39e5dcd6",
  "math_dec_step.onnx": "5ff6e6cbaf0dea70314f66222a1469736a8c5e7994fab06a5b4e53c90469239f",
  "math_enc.onnx": "d4e4a6c40dad1ee988d32c6403b1590ca42864d4483bbe1e671c5dbb999a5a97",
  "math_meta.json": "85d9715fb8a77791b55ef89f57fb41e72eb3170ec58a7c3bee3b394268bf2b10",
  "mathcls.onnx": "818b1556e9ef337958db6be0c9791299d3f74a4575586b0cd469aaae9534ff96",
  "meta.json": "ff6ec58f91c93a475ae9505ad21c73c3ecda0b378f1d889dd83408e10832d1bb",
  "order.onnx": "23635dd2324b4ccc00c408af961273a2dedd2e4f0386bc54dba678c41bcf6b46",
  "orient.onnx": "8d7bfce5d08bcad55955f41ca494a7f9d9d3ed5d5b6da1f6f276d5c509590729",
  "rec.onnx": "e243be069b357e56a0e867a21d7482e498df42fb19678bfb67a25f48b1f06151",
  "rec_cjk.onnx": "4cec840b55e1bb7f375daf30a2975404a867a71ac1a5bc5b1629c4909ce16baa",
  "script.onnx": "5ba37030e817affede0097a7503b6829624104166c6e0b6aacecacf919fc7d11",
} satisfies Record<string, string>;

const UPSTREAM_SHA256_V1_1 = {
  LICENSE: "50c88a018a7ee322e89f6b09401b237beb0cd331685879405b3db48993950a5f",
  "det.onnx": "f2bd4cba23e9c328cf7ec824634e423e191a2c6aacbbf35c62bd79dd70f62f59",
  "lm.npz": "6c0fb618a363a0acdd05d4f5b0edd9e8146d2d990541b87f6130bbfe17cea58d",
  "lm_cjk.npz": "6d64f13f4f90f4e529bd63340c6df76238d007feaef2abb903d8cd28d92bc2f3",
  "lm_world.npz": "34df5e74a2fcbb9cf5e7f360a559375ffa2e04d2d2943a42308c177bcb444d5d",
  "math_dec_init.onnx": "919417fde9962c3318aeb478c1fe3955dd2bb8cd34476c68debed2ba39e5dcd6",
  "math_dec_step.onnx": "5ff6e6cbaf0dea70314f66222a1469736a8c5e7994fab06a5b4e53c90469239f",
  "math_enc.onnx": "d4e4a6c40dad1ee988d32c6403b1590ca42864d4483bbe1e671c5dbb999a5a97",
  "math_meta.json": "85d9715fb8a77791b55ef89f57fb41e72eb3170ec58a7c3bee3b394268bf2b10",
  "mathcls.onnx": "818b1556e9ef337958db6be0c9791299d3f74a4575586b0cd469aaae9534ff96",
  "meta.json": "4ce4c94b46f5d644033c02e718322014ff4081270f385038c571912b56696339",
  "order.onnx": "a42db2e8b536ad3d84045a4b7f9b0a72667cf8be4aa02d63f7b38004719fb180",
  "orient.onnx": "20eaf670f55a7cf9facfe1bdd6c7b8e40be75bbf4d48a3df43f66aa3bbd874fd",
  "rec.onnx": "475972a2710be4a3cbfeb2bc7f3624ffd563de07de42476fc7a290d4e1ece484",
  "rec_cjk.onnx": "4a035516a17e62cdf33a4fe2d88077654cb8841cff024685cdef4e351694ef50",
  "rec_rtl.onnx": "468796bd684486a8388c30c8b4ed2a26e3eeb1f8ff02831c24c2e12c79a2a834",
  "rec_world.onnx": "c8d1e59f21b04966f179d2d2d8e22b230d22110b3f89855bbbf2624a58c40043",
  "script.onnx": "b6ce4e926e19c243faaf37bd46880398bf2cb4ae56a29e1177a3b017c1280e36",
} satisfies Record<string, string>;

const LFS_POINTER =
  /^version https:\/\/git-lfs\.github\.com\/spec\/v1\noid sha256:([0-9a-f]{64})\n/;

function sha256Of(path: string): string {
  const bytes = readFileSync(path);
  const pointer = LFS_POINTER.exec(bytes.subarray(0, 200).toString("latin1"));
  if (pointer?.[1]) return pointer[1];
  return createHash("sha256").update(bytes).digest("hex");
}

describe.each([
  ["models/", MODELS_DIR, UPSTREAM_SHA256, ["README.md", "v1.1"]],
  ["models/v1.1/", join(MODELS_DIR, "v1.1"), UPSTREAM_SHA256_V1_1, ["README.md"]],
] as const)("%s mirror", (_, dir, hashes, extra) => {
  test("holds exactly the upstream files plus its README", () => {
    const files = readdirSync(dir).toSorted();
    expect(files).toEqual([...Object.keys(hashes), ...extra].toSorted());
  });

  test.each(Object.entries(hashes))("%s matches upstream", (file, sha256) => {
    expect(sha256Of(join(dir, file))).toBe(sha256);
  });
});
