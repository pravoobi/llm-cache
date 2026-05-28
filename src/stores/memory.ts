import type { StoreAdapter, CacheEntry, EmbeddingRecord } from '../types'
import { isExpired } from '../utils/ttl'

export function memoryStore(): StoreAdapter {
  const entries = new Map<string, CacheEntry>()
  const embeddings: EmbeddingRecord[] = []

  return {
    async get(key: string): Promise<CacheEntry | null> {
      const entry = entries.get(key)
      if (!entry) return null
      // Lazily evict expired entries on read rather than running a background timer.
      if (isExpired(entry)) {
        entries.delete(key)
        return null
      }
      return entry
    },

    async set(key: string, entry: CacheEntry, _ttlSeconds?: number): Promise<void> {
      entries.set(key, entry)

      // Keep a separate flat list of embedding records for similarity search.
      // Replace existing record for the same key so we don't accumulate stale vectors.
      const existingIdx = embeddings.findIndex((r) => r.key === key)
      const record: EmbeddingRecord = {
        key,
        embedding: entry.embedding,
        createdAt: entry.createdAt,
        ...(entry.namespace !== undefined ? { namespace: entry.namespace } : {}),
      }
      if (existingIdx >= 0) {
        embeddings[existingIdx] = record
      } else {
        embeddings.push(record)
      }
    },

    async delete(key: string): Promise<void> {
      entries.delete(key)
      const idx = embeddings.findIndex((r) => r.key === key)
      if (idx >= 0) embeddings.splice(idx, 1)
    },

    async listEmbeddings(namespace?: string): Promise<EmbeddingRecord[]> {
      if (namespace === undefined) return [...embeddings]
      return embeddings.filter((r) => r.namespace === namespace)
    },
  }
}
