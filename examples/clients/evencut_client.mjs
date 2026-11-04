// Minimal client for the Evencut HTTP service (Node.js 18+ or any browser with fetch).
//
//   node evencut_client.mjs                    # uses http://127.0.0.1:8000
//   EVENCUT_URL=http://host:8000 node evencut_client.mjs
//
// JSON numbers are IEEE doubles in JavaScript: keep weights and totals below 2^53.
// sum_of_squares is returned as a string for that reason.

const USER_AGENT = 'evencut-client-js/1.0.0';

export class EvencutAPIError extends Error {
  constructor(status, detail) {
    super(`HTTP ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

/**
 * @typedef {Object} Partition
 * @property {string} objective
 * @property {number} parts
 * @property {number[]} ends   exclusive end index of each part
 * @property {number[]} loads
 * @property {number} max_load
 * @property {number} total
 * @property {string} sum_of_squares
 * @property {number} imbalance
 */

export class EvencutClient {
  constructor(baseUrl = 'http://127.0.0.1:8000', { retries = 3, timeoutMs = 60000 } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.retries = retries;
    this.timeoutMs = timeoutMs;
  }

  async #post(path, payload) {
    const body = JSON.stringify(payload);
    for (let attempt = 0; ; attempt++) {
      let res;
      try {
        res = await fetch(this.baseUrl + path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'User-Agent': USER_AGENT },
          body,
          signal: AbortSignal.timeout(this.timeoutMs),
        });
      } catch (err) {
        if (attempt >= this.retries) throw err;
        await new Promise((r) => setTimeout(r, Math.min(250 * 2 ** attempt, 4000)));
        continue;
      }
      if (res.ok) return res.json();
      const retryable = res.status === 429 || res.status >= 500;
      if (!retryable || attempt >= this.retries) {
        const detail = (await res.json().catch(() => ({}))).detail ?? res.statusText;
        throw new EvencutAPIError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail));
      }
      await new Promise((r) => setTimeout(r, Math.min(250 * 2 ** attempt, 4000)));
    }
  }

  /** @returns {Promise<Partition>} */
  partition(weights, k, objective = 'min_max', options = {}) {
    return this.#post('/v1/partition', { weights, k, objective, ...options });
  }

  pipelineStages(layerCosts, numStages, options = {}) {
    return this.#post('/v1/ai/pipeline-stages', { layer_costs: layerCosts, num_stages: numStages, ...options });
  }
}

// Example usage when run directly with Node (skipped when imported, and in browsers).
const runDirectly = typeof process !== 'undefined' && typeof process.argv?.[1] === 'string'
  && process.argv[1].replace(/\\/g, '/').endsWith('evencut_client.mjs');
if (runDirectly) {
  const client = new EvencutClient(process.env.EVENCUT_URL ?? 'http://127.0.0.1:8000');
  const p = await client.partition([5, 3, 8, 2, 7, 4, 6, 1, 9, 5], 3);
  console.log('partition:', p.ends, p.loads, 'max', p.max_load);
  const plan = await client.pipelineStages(Array(12).fill(1.0), 3, { embedding_cost: 2.0, head_cost: 3.0 });
  console.log('pipeline:', plan.layers_per_stage, plan.megatron_layout);
  try {
    await client.partition([1, 2, 3], 0);
  } catch (e) {
    console.log('error handled:', e.status);
  }
}
