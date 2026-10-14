// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/** An 8-bit grayscale image, row-major, one byte per pixel (0 = black). */
export type Gray = { data: Uint8Array; width: number; height: number };

/** A single-channel float32 image, row-major. */
export type FloatImage = { data: Float32Array; width: number; height: number };

/** A 2x3 affine matrix, row-major: [a, b, c, d, e, f] maps (x, y) to (ax+by+c, dx+ey+f). */
export type Affine = [number, number, number, number, number, number];

export type Interpolation = "area" | "linear";

/**
 * The image operations the Indra pipeline needs, with OpenCV semantics
 * (`cv2.resize`, `cv2.warpAffine`). The opencv engine implements them with
 * OpenCV.js and matches the Python reference byte for byte; the canvas-native
 * engine implements them in plain TypeScript.
 */
export type ImageOps = {
  /** `cv2.resize(src, (width, height), interpolation)` on uint8. */
  resize(src: Gray, width: number, height: number, interpolation: Interpolation): Gray;
  /** `cv2.warpAffine(src, m, (width, height), INTER_LINEAR, BORDER_REPLICATE)` on uint8. */
  warpReplicate(src: Gray, m: Affine, width: number, height: number): Gray;
  /** `cv2.warpAffine(src, m, (width, height))` on float32: linear, zero border. */
  warpFloat(src: FloatImage, m: Affine, width: number, height: number): FloatImage;
};

/** `cv2.getRotationMatrix2D(center, degrees, 1.0)`: counter-clockwise about the center. */
export function rotationMatrix(cx: number, cy: number, degrees: number): Affine {
  const rad = (degrees * Math.PI) / 180;
  const a = Math.cos(rad);
  const b = Math.sin(rad);
  return [a, b, (1 - a) * cx - b * cy, -b, a, b * cx + (1 - a) * cy];
}

/** `1 - gray / 255` as float32: the ink map the models read (1 = ink). */
export function inkOf(gray: Gray): FloatImage {
  const data = new Float32Array(gray.data.length);
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.fround(1 - Math.fround((gray.data[i] ?? 0) / 255));
  }
  return { data, width: gray.width, height: gray.height };
}

/**
 * 8-bit gray from RGBA pixels, with OpenCV's fixed-point BT.601 weights
 * (cv.cvtColor RGBA2GRAY): Y = (4899 R + 9617 G + 1868 B + 8192) >> 14.
 * Alpha is ignored, as cv2.imread ignores it.
 */
export function grayFromRgba(rgba: {
  data: ArrayLike<number>;
  width: number;
  height: number;
}): Gray {
  const n = rgba.width * rgba.height;
  const data = new Uint8Array(n);
  const d = rgba.data;
  for (let i = 0; i < n; i++) {
    const p = i * 4;
    data[i] = ((d[p] ?? 0) * 4899 + (d[p + 1] ?? 0) * 9617 + (d[p + 2] ?? 0) * 1868 + 8192) >> 14;
  }
  return { data, width: rgba.width, height: rgba.height };
}
