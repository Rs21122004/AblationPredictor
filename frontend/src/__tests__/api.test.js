import { describe, expect, it, vi } from 'vitest';

import { checkHealth } from '../services/api';

describe('api service', () => {
  it('calls health endpoint', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'healthy' }),
    });
    const result = await checkHealth();
    expect(result.status).toBe('healthy');
  });
});
