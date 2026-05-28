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
