// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { rmSync } from "fs";

import { CACHE_DIR } from "../src/processor/model-cache.js";

rmSync(CACHE_DIR, { recursive: true, force: true });
console.log(`Removed ${CACHE_DIR}`);
