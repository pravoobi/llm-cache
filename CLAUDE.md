# llm-cache

A semantic caching layer for LLM calls that deduplicates near-identical prompts using embeddings, saving cost and latency. Provider-agnostic, store-agnostic, TypeScript-first, zero required dependencies beyond optional peer deps.

---

## Project goal

Build and publish a production-ready npm package (`llm-cache`) that:
- Intercepts LLM calls and returns cached responses for semantically similar prompts
- Uses a dual-layer strategy: exact hash match first (free), then semantic similarity (cheap embedding call), then real LLM call (full cost)
- Is fully pluggable: bring your own embedder and storage backend
- Works in Node.js 18+ out of the box with zero mandatory external services

---

## Repository structure

```
llm-cache/
├── src/
│   ├── index.ts                  # Public API exports
│   ├── cache.ts                  # Core createCache() factory + wrap() logic
│   ├── hash.ts                   # Exact-match SHA-256 hashing
│   ├── similarity.ts             # Cosine similarity computation
│   ├── types.ts                  # All shared TypeScript interfaces
│   ├── embedders/
│   │   ├── index.ts              # Embedder registry + factory
│   │   ├── openai.ts             # OpenAI text-embedding-3-small/large
│   │   ├── anthropic.ts          # Anthropic voyage-3 embeddings
│   │   ├── cohere.ts             # Cohere embed-v3
│   │   └── local.ts              # Local ONNX model (optional peer dep)
│   ├── stores/
│   │   ├── index.ts              # Store registry + factory
│   │   ├── memory.ts             # In-memory store (default, no deps)
│   │   ├── redis.ts              # Redis store (ioredis peer dep)
│   │   ├── sqlite.ts             # SQLite store (better-sqlite3 peer dep)
│   │   └── pgvector.ts           # Postgres pgvector store (pg peer dep)
│   └── utils/
│       ├── namespace.ts          # Cache key namespacing helpers
│       └── ttl.ts                # TTL expiry helpers
├── test/
│   ├── cache.test.ts
│   ├── hash.test.ts
│   ├── similarity.test.ts
│   ├── embedders/
│   │   └── openai.test.ts
│   └── stores/
│       ├── memory.test.ts
│       └── redis.test.ts
├── examples/
│   ├── basic.ts                  # Simple in-memory example
│   ├── with-redis.ts             # Redis backend example
│   └── with-openai.ts            # OpenAI embedder example
├── package.json
├── tsconfig.json
├── tsconfig.build.json
├── vitest.config.ts
├── .eslintrc.json
├── .prettierrc
└── README.md
```

---

## Core types (`src/types.ts`)

```ts
export type EmbedFn = (text: string) => Promise<Float32Array | number[]>

export interface EmbedderConfig {
  provider: "openai" | "anthropic" | "cohere" | "local"
  model?: string
  apiKey?: string
}

export interface StoreAdapter {
  get(key: string): Promise<CacheEntry | null>
  set(key: string, entry: CacheEntry, ttlSeconds?: number): Promise<void>
  delete(key: string): Promise<void>
  listEmbeddings(namespace?: string): Promise<EmbeddingRecord[]>
  close?(): Promise<void>
}

export interface EmbeddingRecord {
  key: string
  embedding: number[]
  namespace?: string
  createdAt: number
}

export interface CacheEntry {
  prompt: string
  response: unknown
  embedding: number[]
  namespace?: string
  createdAt: number
  expiresAt?: number
}

export interface CacheOptions {
  /** Cosine similarity threshold for a cache hit. Default: 0.92 */
  threshold?: number
  /** TTL in seconds. Default: 86400 (24h). Set 0 to disable expiry. */
  ttl?: number
  /** Namespace to scope cache lookups (e.g. per-user, per-session) */
  namespace?: string
  /** Include system prompt in cache key to prevent cross-context hits */
  context?: string
  /** Force bypass the cache for this call */
  bypass?: boolean
}

export interface CacheResult<T> {
  value: T
  hit: boolean
  layer: "exact" | "semantic" | "miss"
  similarity?: number
  matchedPrompt?: string
  namespace?: string
}

export interface LLMCacheConfig {
  embedder: EmbedderConfig | EmbedFn
  store?: StoreAdapter | "memory"
  threshold?: number
  ttl?: number
  onHit?: (result: CacheResult<unknown>) => void
  onMiss?: (prompt: string) => void
  onError?: (err: Error) => void
}
```

---

## Core logic (`src/cache.ts`)

### `createCache(config)`

Returns a cache instance with a single primary method: `wrap()`.

### `cache.wrap(prompt, fn, options?)`

```
1. Normalize prompt (trim, collapse whitespace)
2. Build cache key = SHA-256(namespace + context + normalizedPrompt)
3. Check store for exact key match → return CacheResult { hit: true, layer: "exact" }
4. Generate embedding for normalizedPrompt via embedder
5. Load all EmbeddingRecords from store (filtered by namespace if set)
6. Compute cosine similarity between new embedding and each stored embedding
7. If best similarity >= threshold → fetch full entry, return CacheResult { hit: true, layer: "semantic", similarity, matchedPrompt }
8. Call fn() to get real LLM response
9. Store new CacheEntry (prompt, response, embedding, namespace, timestamps)
10. Return CacheResult { hit: false, layer: "miss" }
```

### `cache.invalidate(prompt, options?)`

Remove a specific cache entry by prompt.

### `cache.flush(namespace?)`

Clear all entries, or all entries within a namespace.

### `cache.stats()`

Return `{ totalEntries, namespaces, hitRate (lifetime), avgSimilarity }`.

---

## Dual-layer strategy

```
Incoming prompt
      │
      ▼
 Exact hash match? ──yes──▶ return cached response (zero cost)
      │ no
      ▼
 Generate embedding (embedding API call — ~0.01¢)
      │
      ▼
 Similarity search ──hit──▶ return cached response (embedding cost only)
      │ miss
      ▼
 Call LLM fn()     (full LLM cost)
      │
      ▼
 Store entry + embedding for future hits
```

---

## Embedder implementations

### OpenAI (`src/embedders/openai.ts`)
- Default model: `text-embedding-3-small`
- Uses `openai` peer dep or raw fetch fallback
- Normalize output to `Float32Array`

### Anthropic (`src/embedders/anthropic.ts`)
- Uses Voyage AI via Anthropic SDK: `voyage-3`
- Peer dep: `@anthropic-ai/sdk`

### Cohere (`src/embedders/cohere.ts`)
- Model: `embed-english-v3.0`
- Input type: `search_document` for storing, `search_query` for lookup

### Local (`src/embedders/local.ts`)
- Uses `@xenova/transformers` (peer dep) with `all-MiniLM-L6-v2`
- Fully offline — no API key needed
- Slower on first call (model download), cached thereafter

### Custom embedder
If `embedder` is a function instead of a config object, call it directly:
```ts
embedder: async (text) => myModel.embed(text) // must return number[] or Float32Array
```

---

## Store implementations

### Memory (`src/stores/memory.ts`)
- Default store, zero deps
- `Map<string, CacheEntry>` for entries
- Array of `EmbeddingRecord` for similarity search
- TTL enforced via `expiresAt` timestamp check on read
- Not persistent across process restarts

### Redis (`src/stores/redis.ts`)
- Peer dep: `ioredis`
- Entries stored as JSON strings with Redis TTL (`SETEX`)
- Embeddings stored in a sorted set per namespace for fast retrieval
- Use `SCAN` to avoid blocking on large datasets

### SQLite (`src/stores/sqlite.ts`)
- Peer dep: `better-sqlite3`
- Table: `cache_entries (key TEXT PRIMARY KEY, entry JSON, expires_at INTEGER)`
- Table: `cache_embeddings (key TEXT, namespace TEXT, embedding BLOB, created_at INTEGER)`
- Index on `namespace` and `expires_at`
- Good for single-process, persistent, no-infra use cases

### Postgres/pgvector (`src/stores/pgvector.ts`)
- Peer dep: `pg`
- Uses `pgvector` extension for native ANN similarity search
- Table: `llm_cache (key TEXT, prompt TEXT, response JSONB, embedding vector(1536), namespace TEXT, created_at TIMESTAMPTZ, expires_at TIMESTAMPTZ)`
- Index: `CREATE INDEX ON llm_cache USING ivfflat (embedding vector_cosine_ops)`
- Best for multi-process, high-traffic production use

---

## Similarity computation (`src/similarity.ts`)

```ts
export function cosineSimilarity(a: number[], b: number[]): number
export function findBestMatch(
  query: number[],
  records: EmbeddingRecord[],
  threshold: number
): { record: EmbeddingRecord; similarity: number } | null
```

- Pure implementation, no external deps
- Handle zero-vector edge case (return 0)
- For memory store with >10k entries, warn in console and suggest pgvector

---

## Namespace + context scoping (`src/utils/namespace.ts`)

Cache keys must be scoped correctly to avoid false positives:

```ts
buildCacheKey(prompt: string, namespace?: string, context?: string): string
// Returns SHA-256 of: `${namespace ?? ""}:${context ?? ""}:${normalizedPrompt}`
```

Embeddings are also tagged with namespace so similarity search only compares entries in the same namespace.

---

## Public API (`src/index.ts`)

Export the following:

```ts
export { createCache } from './cache'
export type {
  LLMCacheConfig,
  CacheOptions,
  CacheResult,
  StoreAdapter,
  EmbedFn,
  EmbedderConfig,
  CacheEntry,
  EmbeddingRecord,
} from './types'

// Store adapters (for custom setup)
export { memoryStore }   from './stores/memory'
export { redisStore }    from './stores/redis'
export { sqliteStore }   from './stores/sqlite'
export { pgvectorStore } from './stores/pgvector'
```

---

## package.json

```json
{
  "name": "llm-cache",
  "version": "0.1.0",
  "description": "Semantic caching layer for LLM calls. Deduplicates near-identical prompts using embeddings.",
  "main": "./dist/index.js",
  "module": "./dist/index.mjs",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "import": "./dist/index.mjs",
      "require": "./dist/index.js",
      "types": "./dist/index.d.ts"
    }
  },
  "files": ["dist"],
  "scripts": {
    "build": "tsup src/index.ts --format cjs,esm --dts",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "eslint src --ext .ts",
    "typecheck": "tsc --noEmit"
  },
  "peerDependencies": {
    "openai": ">=4.0.0",
    "@anthropic-ai/sdk": ">=0.20.0",
    "ioredis": ">=5.0.0",
    "better-sqlite3": ">=9.0.0",
    "pg": ">=8.0.0",
    "@xenova/transformers": ">=2.0.0"
  },
  "peerDependenciesMeta": {
    "openai": { "optional": true },
    "@anthropic-ai/sdk": { "optional": true },
    "ioredis": { "optional": true },
    "better-sqlite3": { "optional": true },
    "pg": { "optional": true },
    "@xenova/transformers": { "optional": true }
  },
  "devDependencies": {
    "typescript": "^5.4.0",
    "tsup": "^8.0.0",
    "vitest": "^1.6.0",
    "@types/node": "^20.0.0",
    "@types/better-sqlite3": "^7.0.0",
    "@types/pg": "^8.0.0",
    "eslint": "^8.0.0",
    "@typescript-eslint/parser": "^7.0.0",
    "@typescript-eslint/eslint-plugin": "^7.0.0"
  },
  "keywords": ["llm", "cache", "semantic", "embeddings", "openai", "anthropic", "ai"],
  "license": "MIT",
  "engines": { "node": ">=18.0.0" }
}
```

---

## tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "exactOptionalPropertyTypes": true,
    "noUncheckedIndexedAccess": true,
    "declaration": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "skipLibCheck": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist", "test"]
}
```

---

## Testing strategy

Use **vitest**. Mock all external API calls.

### Unit tests (no network)
- `hash.test.ts` — deterministic keys for same/different inputs
- `similarity.test.ts` — cosine similarity correctness, edge cases (zero vector, identical vectors)
- `stores/memory.test.ts` — get/set/delete/TTL expiry/namespace filtering
- `cache.test.ts` — full wrap() flow with mocked embedder and memory store

### Integration stubs
- `embedders/openai.test.ts` — mock fetch, assert correct request shape, assert output normalization

### Key test cases for `cache.test.ts`
1. Exact same prompt → hits exact layer, fn() never called
2. Semantically similar prompt above threshold → hits semantic layer, fn() never called
3. Different prompt below threshold → misses, fn() called, entry stored
4. `bypass: true` → always calls fn(), never reads or writes cache
5. Different namespaces → no cross-namespace hits
6. Different context strings → no cross-context hits
7. TTL expiry → expired entry treated as miss
8. fn() throws → error propagates, nothing cached
9. Embedder throws → error propagates via onError, fn() called as fallback (graceful degradation)

---

## Error handling principles

- **Never let cache failures break the LLM call.** If the store or embedder throws, log via `onError`, skip the cache, and call `fn()` directly.
- **Throw only on invalid config** (e.g. unknown provider, missing API key at init time).
- All async errors caught internally in `wrap()` with try/catch.

---

## Implementation notes

- Normalize prompts before hashing and embedding: `prompt.trim().replace(/\s+/g, ' ')`
- Always store both the hash key (for exact match) and the embedding record (for semantic match) in the same `set()` call
- The memory store similarity search is O(n) — acceptable up to ~50k entries; beyond that the README should recommend pgvector
- Cosine similarity requires both vectors to be the same dimension — validate at runtime and throw a descriptive error if they differ (indicates embedder model was changed mid-cache)
- `stats()` tracks hits/misses in memory on the cache instance (not persisted)

---

## README must include

1. Install instructions + peer dep install examples for each provider
2. Quick start (5-line example with in-memory store)
3. Full API reference for `createCache`, `wrap`, `invalidate`, `flush`, `stats`
4. All embedder configs with example
5. All store adapter setups with example
6. Namespace + context scoping explanation with false-positive warning
7. Cost savings example: "~33% of LLM queries are semantically duplicated — at $10/1M tokens, 100k queries/day saves ~$X"
8. When NOT to use semantic caching (dynamic data, real-time queries, highly personalized responses)
9. Migration guide from exact-match to semantic cache

---

## What NOT to build

- No CLI
- No server / HTTP layer (that's Bifrost's job)
- No prompt management (that's `prompt-template`'s job)
- No streaming support in v0.1 (log a clear error if a streaming response is passed to wrap())
- No vector index (ANN) in memory store — linear scan is fine for v0.1; pgvector handles scale
