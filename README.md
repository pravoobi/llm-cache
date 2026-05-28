# @pravoobi/llm-cache

Semantic caching layer for LLM calls. Deduplicates near-identical prompts using embeddings, saving cost and latency. Provider-agnostic, store-agnostic, TypeScript-first, zero required dependencies.

## How it works

```
Incoming prompt
      │
      ▼
 Exact hash match? ──yes──▶ return cached response   (zero cost)
      │ no
      ▼
 Generate embedding              (~$0.0001 per call)
      │
      ▼
 Cosine similarity search ──hit──▶ return cached response
      │ miss
      ▼
 Call your LLM fn()                   (full cost)
      │
      ▼
 Store entry + embedding for future hits
```

Exact matches are free (SHA-256 hash lookup). Semantic matches cost one embedding call — roughly 1000× cheaper than re-running the LLM. Cache failures are always non-fatal: if the store or embedder throws, `wrap()` falls back to calling your function directly.

---

## Install

```bash
npm install @pravoobi/llm-cache
```

Install the peer dep for your chosen embedder:

```bash
# OpenAI
npm install openai

# Anthropic (Voyage AI)
npm install @anthropic-ai/sdk

# Cohere
# (no SDK needed — uses raw fetch)

# Local / offline
npm install @xenova/transformers
```

Install the peer dep for your chosen store (optional — in-memory is the default):

```bash
# Redis
npm install ioredis

# SQLite
npm install better-sqlite3

# Postgres / pgvector
npm install pg
```

---

## Quick start

```ts
import { createCache } from '@pravoobi/llm-cache'

const cache = createCache({
  embedder: { provider: 'openai', apiKey: process.env.OPENAI_API_KEY },
})

const result = await cache.wrap(
  'What is the capital of France?',
  () => myLLM.call('What is the capital of France?')
)

console.log(result.value)   // 'Paris is the capital of France.'
console.log(result.hit)     // true on second call with same/similar prompt
console.log(result.layer)   // 'exact' | 'semantic' | 'miss'
```

---

## Full API reference

### `createCache(config)`

```ts
import { createCache } from '@pravoobi/llm-cache'

const cache = createCache({
  embedder: { provider: 'openai', apiKey: '...' }, // or a custom EmbedFn
  store: memoryStore(),    // default — see Store adapters below
  threshold: 0.92,         // cosine similarity threshold (default: 0.9)
  ttl: 86400,              // TTL in seconds; 0 = no expiry (default: none)
  onHit:  (result) => {},  // called on every cache hit
  onMiss: (prompt) => {},  // called on every cache miss
  onError: (err) => {},    // called on non-fatal cache errors
})
```

---

### `cache.wrap(prompt, fn, options?)`

The primary method. Returns a `CacheResult<T>`.

```ts
const result = await cache.wrap(
  prompt,                // string — the prompt to cache
  () => myLLM(prompt),  // () => Promise<T> — your LLM call
  {
    threshold: 0.95,    // override per-call similarity threshold
    ttl: 3600,          // override per-call TTL
    namespace: 'user-42', // scope cache to a namespace
    context: 'sys-v2',  // include system prompt in cache key
    bypass: false,      // set true to skip cache entirely
  }
)
```

**`CacheResult<T>`**:

| Field | Type | Description |
|---|---|---|
| `value` | `T` | The response (cached or live) |
| `hit` | `boolean` | Whether it was served from cache |
| `layer` | `"exact" \| "semantic" \| "miss"` | Which cache layer matched |
| `similarity` | `number?` | Cosine similarity score (semantic hits only) |
| `matchedPrompt` | `string?` | The original prompt that was matched (semantic hits only) |
| `namespace` | `string?` | The namespace used for this call |

> **Streaming is not supported.** If `fn()` returns a `ReadableStream` or async iterable, `wrap()` will throw. Collect the full response before passing it to `wrap()`, or use `bypass: true`.

---

### `cache.invalidate(prompt, options?)`

Remove a specific entry from the cache.

```ts
await cache.invalidate('What is the capital of France?', { namespace: 'user-42' })
```

---

### `cache.flush(namespace?)`

Clear all entries, or all entries within a namespace.

```ts
await cache.flush()           // clear everything
await cache.flush('user-42')  // clear only user-42's entries
```

---

### `cache.stats()`

Lifetime statistics for this cache instance (not persisted across restarts).

```ts
const { totalEntries, namespaces, hitRate, avgSimilarity } = cache.stats()
```

| Field | Type | Description |
|---|---|---|
| `totalEntries` | `number` | Total calls (hits + misses) |
| `namespaces` | `string[]` | All namespaces seen this session |
| `hitRate` | `number` | Fraction of calls served from cache (0–1) |
| `avgSimilarity` | `number` | Average cosine similarity of semantic hits |

---

## Embedder configs

### OpenAI

```ts
createCache({
  embedder: {
    provider: 'openai',
    model: 'text-embedding-3-small', // default
    apiKey: process.env.OPENAI_API_KEY,
  },
})
```

### Anthropic (Voyage AI)

```ts
createCache({
  embedder: {
    provider: 'anthropic',
    model: 'voyage-3', // default
    apiKey: process.env.VOYAGE_API_KEY,
  },
})
```

### Cohere

```ts
createCache({
  embedder: {
    provider: 'cohere',
    model: 'embed-english-v3.0', // default
    apiKey: process.env.COHERE_API_KEY,
  },
})
```

### Local (offline)

```ts
// Requires: npm install @xenova/transformers
createCache({
  embedder: {
    provider: 'local',
    model: 'Xenova/all-MiniLM-L6-v2', // default — downloads on first call
  },
})
```

### Custom embedder

```ts
createCache({
  embedder: async (text: string) => {
    const vec = await myModel.embed(text)
    return vec // number[] or Float32Array
  },
})
```

---

## Store adapter setups

### In-memory (default)

```ts
import { createCache, memoryStore } from '@pravoobi/llm-cache'

createCache({ embedder: ..., store: memoryStore() })
// or just omit `store` — memory is the default
```

Not persistent across restarts. Suitable for single-process, development, or short-lived workloads.

### Redis

```ts
import Redis from 'ioredis'
import { createCache, redisStore } from '@pravoobi/llm-cache'

const client = new Redis({ host: 'localhost', port: 6379 })
createCache({ embedder: ..., store: redisStore(client) })
```

### SQLite

```ts
import Database from 'better-sqlite3'
import { createCache, sqliteStore } from '@pravoobi/llm-cache'

const db = new Database('./cache.db')
createCache({ embedder: ..., store: sqliteStore(db) })
```

Good for single-process, persistent, no-infra setups.

### Postgres / pgvector

```ts
import { Pool } from 'pg'
import { createCache, pgvectorStore } from '@pravoobi/llm-cache'

const pool = new Pool({ connectionString: process.env.DATABASE_URL })
createCache({ embedder: ..., store: pgvectorStore(pool) })
```

Requires the [`pgvector`](https://github.com/pgvector/pgvector) Postgres extension. Best for multi-process, high-traffic production use. Uses native ANN similarity search via `ivfflat`.

---

## Namespace and context scoping

**Namespaces** isolate cache entries so they are never shared across different users, tenants, or sessions:

```ts
// user-1 and user-2 will never get each other's cached responses
await cache.wrap(prompt, fn, { namespace: `user-${userId}` })
```

**Context** embeds your system prompt into the cache key, preventing a prompt that was answered under one system prompt from being served under a different one:

```ts
await cache.wrap(prompt, fn, { context: systemPrompt })
```

> **False-positive risk:** Without namespacing, two users asking semantically identical questions will share a cached response. This is usually desirable for factual queries but problematic for personalized or private data. Use `namespace` to enforce isolation wherever user data is involved.

Both namespace and context are combined into the embedding scope, so semantic similarity search is always bounded within the same namespace + context pair.

---

## Cost savings

Studies show roughly 30–40% of LLM queries in production are semantically near-duplicate. At $10 per million tokens with an average prompt of 500 tokens:

| Daily queries | Cache hit rate | Daily savings |
|---|---|---|
| 10,000 | 30% | ~$15 |
| 100,000 | 30% | ~$150 |
| 1,000,000 | 30% | ~$1,500 |

Embedding costs (e.g., `text-embedding-3-small` at $0.02/million tokens) are negligible by comparison — a 500-token prompt costs ~$0.00001 to embed versus ~$0.005 to run through GPT-4o.

---

## When NOT to use semantic caching

- **Real-time or live data queries** — "What is the current stock price of AAPL?" should never be cached.
- **Highly personalized responses** — If the correct answer genuinely depends on who is asking, use per-user namespaces carefully or disable caching.
- **Creative or stochastic tasks** — Caching "Write me a poem about autumn" means every user gets the same poem.
- **Short TTLs with fast-changing data** — If your data changes faster than your TTL, stale hits cause more harm than cost savings justify.
- **Streaming responses** — Not supported in v0.1. Collect the full response first.

---

## Migration: from exact-match to semantic cache

If you have an existing exact-match (key/value) cache keyed on the raw prompt string, migrating is straightforward:

1. Replace `cache.get(prompt)` / `cache.set(prompt, response)` with `cache.wrap(prompt, fn)`.
2. Set `threshold: 1.0` initially — this disables semantic matching and gives identical behaviour to exact-match.
3. Lower `threshold` incrementally (e.g., 0.97, 0.95, 0.92) while monitoring `avgSimilarity` from `cache.stats()` to tune precision vs. recall.
4. Add namespaces where you previously had per-user cache key prefixes.
