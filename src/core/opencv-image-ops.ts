// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import type { cv as CvNamespace } from "ppu-ocv";

import type { Affine, Gray, ImageOps, Interpolation } from "../indra/image.js";

type Cv = typeof CvNamespace;

/**
 * ImageOps on OpenCV.js. The reference runs the same OpenCV release (5.0.0)
 * through cv2, so these calls give the same bytes as the Python pipeline.
 * Every Mat is deleted before return: OpenCV.js memory is not garbage collected.
 */
export function opencvImageOps(cv: Cv): ImageOps {
  const interp = (i: Interpolation): number => (i === "area" ? cv.INTER_AREA : cv.INTER_LINEAR);

  const grayMat = (src: Gray): CvNamespace.Mat => {
    const mat = new cv.Mat(src.height, src.width, cv.CV_8UC1);
    mat.data.set(src.data);
    return mat;
  };

  const affineMat = (m: Affine): CvNamespace.Mat => cv.matFromArray(2, 3, cv.CV_64FC1, m);

  return {
    resize(src, width, height, interpolation) {
      const from = grayMat(src);
      const to = new cv.Mat();
      try {
        cv.resize(from, to, new cv.Size(width, height), 0, 0, interp(interpolation));
        return { data: new Uint8Array(to.data), width, height };
      } finally {
        from.delete();
        to.delete();
      }
    },

    warpReplicate(src, m, width, height) {
      const from = grayMat(src);
      const to = new cv.Mat();
      const mat = affineMat(m);
      try {
        cv.warpAffine(
          from,
          to,
          mat,
          new cv.Size(width, height),
          cv.INTER_LINEAR,
          cv.BORDER_REPLICATE
        );
        return { data: new Uint8Array(to.data), width, height };
      } finally {
        from.delete();
        to.delete();
        mat.delete();
      }
    },

    warpZero(src, m, width, height) {
      const from = grayMat(src);
      const to = new cv.Mat();
      const mat = affineMat(m);
      try {
        cv.warpAffine(from, to, mat, new cv.Size(width, height));
        return { data: new Uint8Array(to.data), width, height };
      } finally {
        from.delete();
        to.delete();
        mat.delete();
      }
    },
  };
}
