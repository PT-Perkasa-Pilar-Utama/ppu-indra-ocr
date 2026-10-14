// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 PT Perkasa Pilar Utama

/** Run one operation with a newly initialized service, then release it. */
export async function runOneShot<
  TService extends { initialize(): Promise<void>; destroy(): Promise<void> },
  TResult,
>(createService: () => TService, run: (service: TService) => Promise<TResult>): Promise<TResult> {
  const service = createService();
  try {
    await service.initialize();
    return await run(service);
  } finally {
    await service.destroy();
  }
}
