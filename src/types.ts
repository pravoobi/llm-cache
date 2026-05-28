export type EmbedFn = (text: string) => Promise<Float32Array | number[]>

export interface EmbedderConfig {
  provider: 'openai' | 'anthropic' | 'cohere' | 'local'
  model?: string
  apiKey?: string
}

export interface StoreAdapter {
  get(key: string): Promise<CacheEntry | null>
  set(key: string, entry: CacheEntry, ttlSeconds?: number): Promise<void>
  delete(key: string): Promise<void>
  listEmbeddings(namespace?: string): Promise<EmbeddingRecord[]>
  close?(): Promise<void>
  // Optional ANN search. When present, cache.ts uses it instead of the
  // O(n) listEmbeddings + findBestMatch fallback.
  searchSimilar?(
    query: number[],
    threshold: number,
    namespace?: string
  ): Promise<{ record: EmbeddingRecord; similarity: number } | null>
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
  threshold?: number
  ttl?: number
  namespace?: string
  context?: string
  bypass?: boolean
}

export interface CacheResult<T> {
  value: T
  hit: boolean
  layer: 'exact' | 'semantic' | 'miss'
  similarity?: number
  matchedPrompt?: string
  namespace?: string
}

export interface LLMCacheConfig {
  embedder: EmbedderConfig | EmbedFn
  store?: StoreAdapter | 'memory'
  threshold?: number
  ttl?: number
  onHit?: (result: CacheResult<unknown>) => void
  onMiss?: (prompt: string) => void
  onError?: (err: Error) => void
}

export interface CacheStreamOptions<T> extends CacheOptions {
  // Combines all yielded chunks into the value stored in cache on a miss.
  // Default: joins chunks with '' if they are strings, otherwise stores the array as-is.
  assemble?: (chunks: T[]) => unknown
  // Rebuilds an async iterable from the cached value on a hit.
  // Default: single-item iterable yielding the cached value cast to T.
  reconstruct?: (cached: unknown) => AsyncIterable<T>
}

// Hit/miss metadata returned via the `result` promise from wrapStream().
// No `value` field — the value was already consumed chunk-by-chunk from the stream.
export interface StreamCacheResult {
  hit: boolean
  layer: 'exact' | 'semantic' | 'miss'
  similarity?: number
  matchedPrompt?: string
  namespace?: string
}
