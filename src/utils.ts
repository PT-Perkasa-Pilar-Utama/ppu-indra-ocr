// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

import { mirrorUrl } from "./model-catalogue.js";

/**
 * Ceiling on an honoured `Retry-After`, so a single header cannot stall a
 * download for minutes. A host asking for longer than this gets one more
 * attempt after the cap instead of a wait nobody can interrupt.
 */
const MAX_RETRY_AFTER_MS = 60_000;

/**
 * Parse a `Retry-After` header into a delay in milliseconds.
 *
 * Accepts both forms the spec allows: delta-seconds (`120`) and an HTTP-date
 * (`Wed, 21 Oct 2015 07:28:00 GMT`). Returns null when the header is absent or
 * unparseable, which tells the caller to fall back to its own backoff.
 *
 * @param header - Raw header value, or null when the response carried none.
 * @param now - Current epoch milliseconds, used to resolve the HTTP-date form.
 */
export function parseRetryAfter(header: string | null, now: number): number | null {
  if (!header) return null;
  const trimmed = header.trim();
  if (trimmed === "") return null;

  const seconds = Number(trimmed);
  if (Number.isFinite(seconds)) {
    return seconds <= 0 ? 0 : Math.min(seconds * 1000, MAX_RETRY_AFTER_MS);
  }

  const at = Date.parse(trimmed);
  if (Number.isNaN(at)) return null;
  return Math.min(Math.max(at - now, 0), MAX_RETRY_AFTER_MS);
}

/** Exponential backoff with full jitter: 0.5-1 s, then 1-2 s, then 2-4 s. */
function backoffMs(attempt: number): number {
  const base = 500 * 2 ** attempt;
  return base + Math.random() * base;
}

/**
 * Fetches a URL as an `ArrayBuffer` with a per-attempt deadline and bounded
 * retries. Each attempt is aborted after `timeoutMs` (covering both the
 * response headers and the body download), so a stalled connection fails fast
 * and is retried instead of hanging indefinitely.
 *
 * A failed attempt waits for the response's `Retry-After` header when the host
 * sent one, and otherwise backs off exponentially with jitter.
 *
 * Once every attempt on `url` has failed, the whole sequence runs again against
 * `fallbackUrl` if there is one. By default that is the mirror of a built-in
 * model URL, which is off unless `PPU_INDRA_OCR_MODEL_MIRROR` is set.
 *
 * @param url - Resource to download.
 * @param options - `timeoutMs` per-attempt deadline (default 300 000 ms / 5 min),
 *   `retries` additional attempts after the first (default 2), and `fallbackUrl`
 *   to try once the primary is exhausted (null disables it).
 * @returns The downloaded bytes.
 * @throws If every attempt fails (network error, timeout, or non-2xx response).
 */
export async function fetchArrayBufferWithRetry(
  url: string,
  options: { timeoutMs?: number; retries?: number; fallbackUrl?: string | null } = {}
): Promise<ArrayBuffer> {
  const { timeoutMs = 300_000, retries = 2, fallbackUrl = mirrorUrl(url) } = options;
  let lastError: unknown;

  for (let attempt = 0; attempt <= retries; attempt++) {
    let retryAfterMs: number | null = null;
    try {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(timeoutMs),
        // A model download has no use for a referer, and sending one can cost
        // the download outright: hosts that blocklist an embedding page's
        // origin answer with a status carrying no CORS headers, which the
        // browser then reports as a CORS error rather than as the block it is.
        // Ignored outside browsers, where there is no referer to send.
        referrerPolicy: "no-referrer",
      });
      if (!response.ok) {
        // Read the header before throwing: a rate limiter (HTTP 429) knows how
        // long its window has left, and guessing shorter just burns an attempt.
        retryAfterMs = parseRetryAfter(response.headers.get("retry-after"), Date.now());
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }
      return await response.arrayBuffer();
    } catch (error) {
      lastError = error;
      if (attempt < retries) {
        await new Promise((resolve) => setTimeout(resolve, retryAfterMs ?? backoffMs(attempt)));
      }
    }
  }

  const primaryFailure = new Error(
    `Failed to fetch ${url} after ${retries + 1} attempt(s): ${String(lastError)}`
  );
  if (!fallbackUrl) throw primaryFailure;

  console.warn(`[ppu-indra-ocr] ${primaryFailure.message}\n  retrying on ${fallbackUrl}`);
  try {
    // fallbackUrl: null stops a mirror that is itself unreachable from
    // resolving to a third URL and looping.
    return await fetchArrayBufferWithRetry(fallbackUrl, { timeoutMs, retries, fallbackUrl: null });
  } catch (mirrorError) {
    throw new Error(`${primaryFailure.message}; mirror also failed: ${String(mirrorError)}`, {
      cause: primaryFailure,
    });
  }
}
