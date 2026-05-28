import type { StoreAdapter, CacheEntry, EmbeddingRecord } from '../types'
import { assertCacheEntry } from '../utils/validate'

// Typed minimally to avoid a hard compile-time dep on ioredis.
interface RedisClient {
  get(key: string): Promise<string | null>
  set(key: string, value: string): Promise<unknown>
  setex(key: string, seconds: number, value: string): Promise<unknown>
  del(key: string): Promise<unknown>
  hset(hash: string, field: string, value: string): Promise<unknown>
  hdel(hash: string, field: string): Promise<unknown>
  hgetall(hash: string): Promise<Record<string, string> | null>
  quit(): Promise<unknown>
}

const ENTRY_PREFIX = 'llm-cache:entry:'
const EMBEDDINGS_HASH_PREFIX = 'llm-cache:embeddings:'
const DEFAULT_NS = '__default__'

// Base64url-encode namespace so colons, wildcards, and other Redis key-separator
// characters in user-supplied namespace values cannot traverse the key hierarchy.
function encodeNsSegment(ns: string): string {
  return Buffer.from(ns).toString('base64url')
}

function nsHashKey(namespace: string | undefined): string {
  const segment = namespace !== undefined ? encodeNsSegment(namespace) : DEFAULT_NS
  return `${EMBEDDINGS_HASH_PREFIX}${segment}`
}

export function redisStore(client: unknown): StoreAdapter {
  const redis = client as RedisClient

  return {
    async get(key: string): Promise<CacheEntry | null> {
      const raw = await redis.get(`${ENTRY_PREFIX}${key}`)
      if (!raw) return null
      return assertCacheEntry(JSON.parse(raw), 'redis')
    },

    async set(key: string, entry: CacheEntry, ttlSeconds?: number): Promise<void> {
      const serialized = JSON.stringify(entry)
      const entryKey = `${ENTRY_PREFIX}${key}`

      if (ttlSeconds !== undefined && ttlSeconds > 0) {
        await redis.setex(entryKey, ttlSeconds, serialized)
      } else {
        await redis.set(entryKey, serialized)
      }

      // Store embedding record in a per-namespace hash for fast bulk retrieval.
      const record: EmbeddingRecord = {
        key,
        embedding: entry.embedding,
        createdAt: entry.createdAt,
        ...(entry.namespace !== undefined ? { namespace: entry.namespace } : {}),
      }
      await redis.hset(nsHashKey(entry.namespace), key, JSON.stringify(record))
    },

    async delete(key: string): Promise<void> {
      // We need to remove from both the entry store and potentially multiple
      // namespace hashes. To avoid storing a reverse index, fetch the entry
      // first so we know which namespace hash to clean up.
      const raw = await redis.get(`${ENTRY_PREFIX}${key}`)
      await redis.del(`${ENTRY_PREFIX}${key}`)

      if (raw) {
        const entry = assertCacheEntry(JSON.parse(raw), 'redis')
        await redis.hdel(nsHashKey(entry.namespace), key)
      }
    },

    async listEmbeddings(namespace?: string): Promise<EmbeddingRecord[]> {
      const hash = await redis.hgetall(nsHashKey(namespace))
      if (!hash) return []
      return Object.values(hash).map((v) => JSON.parse(v) as EmbeddingRecord)
    },

    async close(): Promise<void> {
      await redis.quit()
    },
  }
}
