// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { BaseIndraOcrService } from "../core/indra-ocr.service.js";
import type { CoreCanvas } from "../core/platform.js";
import type { IndraOptions } from "../interface.js";
import { MobilePlatformProvider } from "./platform.mobile.js";

/**
 * Indra OCR for React Native (Expo or bare) on iOS and Android, with
 * onnxruntime-react-native and @shopify/react-native-skia. The engine is
 * canvas-native: OpenCV.js does not run on React Native.
 */
export class IndraOcrService extends BaseIndraOcrService<CoreCanvas> {
  public constructor(options: IndraOptions = {}) {
    super(new MobilePlatformProvider(), options);
  }
}
