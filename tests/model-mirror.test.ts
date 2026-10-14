import { describe, expect, test } from "bun:test";
import { createHash } from "crypto";
import { readdirSync, readFileSync } from "fs";
import { join } from "path";

/**
 * models/ mirrors datalab-to/indra-ocr at d7c839c (tag v3.1) unmodified.
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

const LFS_POINTER =
  /^version https:\/\/git-lfs\.github\.com\/spec\/v1\noid sha256:([0-9a-f]{64})\n/;

function sha256Of(path: string): string {
  const bytes = readFileSync(path);
  const pointer = LFS_POINTER.exec(bytes.subarray(0, 200).toString("latin1"));
  if (pointer?.[1]) return pointer[1];
  return createHash("sha256").update(bytes).digest("hex");
}

describe("models/ mirror", () => {
  test("holds exactly the upstream files plus README.md", () => {
    const files = readdirSync(MODELS_DIR).toSorted();
    expect(files).toEqual([...Object.keys(UPSTREAM_SHA256), "README.md"].toSorted());
  });

  test.each(Object.entries(UPSTREAM_SHA256))("%s matches upstream", (file, sha256) => {
    expect(sha256Of(join(MODELS_DIR, file))).toBe(sha256);
  });
});
