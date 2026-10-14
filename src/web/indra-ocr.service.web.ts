// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { BaseIndraOcrService } from "../core/indra-ocr.service.js";
import type { CoreCanvas } from "../core/platform.js";
import type { IndraOptions } from "../interface.js";
import { WebPlatformProvider } from "./platform.web.js";

/**
 * Indra OCR for browsers, web workers and browser extensions. Models are
 * fetched from the default repository (or `models`); images may be encoded
 * bytes or an HTML/Offscreen canvas.
 *
 * The engine defaults to "canvas-native"; pass `engine: "opencv"` to load
 * OpenCV.js and match the reference exactly. ONNX Runtime runs on wasm unless
 * `session.executionProviders` asks for more (WebGPU is untested with these models).
 */
export class IndraOcrService extends BaseIndraOcrService<CoreCanvas> {
  public constructor(options: IndraOptions = {}) {
    super(new WebPlatformProvider(), options);
  }
}
