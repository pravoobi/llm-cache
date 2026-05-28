import type { StoreAdapter, CacheEntry, EmbeddingRecord } from '../types'
import { isExpired } from '../utils/ttl'

// Minimal interface — avoids a compile-time dep on hnswlib-node.
interface HnswIndex {
  initIndex(maxElements: number, efConstruction?: number, m?: number): void
  addPoint(point: number[], label: number): void
  markDelete(label: number): void
  searchKnn(query: number[], k: number): { neighbors: number[]; distances: number[] }
  getCurrentCount(): number
  getMaxElements(): number
  resizeIndex(newSize: number): void
}

export interface HnswLib {
  HierarchicalNSW: new (space: string, dim: number) => HnswIndex
}

const INITIAL_CAPACITY = 1024

// One HNSW index per embedding-namespace so similarity search never crosses
// namespace boundaries and no post-search filtering is needed.
interface NsIndex {
  index: HnswIndex
  keyToLabel: Map<string, number>
  labelToKey: Map<number, string>
  nextLabel: number
  maxElements: number
}

async function loadHnswLib(): Promise<HnswLib> {
  try {
    // Same dynamic-import pattern as local.ts — module string is hardcoded,
    // not user-controlled, so this is not a code injection vector.
    return await (new Function('m', 'return import(m)')('hnswlib-node') as Promise<HnswLib>)
  } catch {
    throw new Error(
      '[llm-cache] hnswMemoryStore requires hnswlib-node: npm install hnswlib-node'
    )
  }
}

export function hnswMemoryStore(
  // Accept a pre-loaded lib instance so tests can inject a mock without
  // needing the native binary installed.
  injectedLib?: HnswLib
): StoreAdapter {
  const entries = new Map<string, CacheEntry>()
  const embeddingRecords = new Map<string, EmbeddingRecord>()
  const nsIndices = new Map<string, NsIndex>()
  let dimension: number | null = null

  // Resolved once on first set(); shared across all subsequent calls.
  let libPromise: Promise<HnswLib> | null = injectedLib ? Promise.resolve(injectedLib) : null

  function getLib(): Promise<HnswLib> {
    if (!libPromise) libPromise = loadHnswLib()
    return libPromise
  }

  function getOrCreateNsIndex(lib: HnswLib, ns: string, dim: number): NsIndex {
    let nsIdx = nsIndices.get(ns)
    if (nsIdx === undefined) {
      const index = new lib.HierarchicalNSW('cosine', dim)
      index.initIndex(INITIAL_CAPACITY)
      nsIdx = { index, keyToLabel: new Map(), labelToKey: new Map(), nextLabel: 0, maxElements: INITIAL_CAPACITY }
      nsIndices.set(ns, nsIdx)
    }
    return nsIdx
  }

  // Embedding-namespace key: undefined namespace stored under a sentinel so
  // Map lookups are consistent.
  function nsKey(namespace: string | undefined): string {
    return namespace ?? '__default__'
  }

  const self: StoreAdapter = {
    async get(key: string): Promise<CacheEntry | null> {
      const entry = entries.get(key)
      if (!entry) return null
      if (isExpired(entry)) {
        await self.delete(key)
        return null
      }
      return entry
    },

    async set(key: string, entry: CacheEntry, _ttlSeconds?: number): Promise<void> {
      const lib = await getLib()

      if (dimension === null) dimension = entry.embedding.length

      const ns = nsKey(entry.namespace)
      const nsIdx = getOrCreateNsIndex(lib, ns, dimension)

      // If this key already exists, soft-delete its old HNSW slot before re-adding.
      const existingLabel = nsIdx.keyToLabel.get(key)
      if (existingLabel !== undefined) {
        try { nsIdx.index.markDelete(existingLabel) } catch { /* already deleted */ }
        nsIdx.labelToKey.delete(existingLabel)
      }

      // Grow index before it fills up.
      if (nsIdx.nextLabel >= nsIdx.maxElements) {
        nsIdx.maxElements *= 2
        nsIdx.index.resizeIndex(nsIdx.maxElements)
      }

      const label = nsIdx.nextLabel++
      nsIdx.index.addPoint(entry.embedding, label)
      nsIdx.keyToLabel.set(key, label)
      nsIdx.labelToKey.set(label, key)

      entries.set(key, entry)
      embeddingRecords.set(key, {
        key,
        embedding: entry.embedding,
        createdAt: entry.createdAt,
        ...(entry.namespace !== undefined ? { namespace: entry.namespace } : {}),
      })
    },

    async delete(key: string): Promise<void> {
      const entry = entries.get(key)
      if (entry) {
        const nsIdx = nsIndices.get(nsKey(entry.namespace))
        if (nsIdx) {
          const label = nsIdx.keyToLabel.get(key)
          if (label !== undefined) {
            try { nsIdx.index.markDelete(label) } catch { /* already deleted */ }
            nsIdx.keyToLabel.delete(key)
            nsIdx.labelToKey.delete(label)
          }
        }
      }
      entries.delete(key)
      embeddingRecords.delete(key)
    },

    async listEmbeddings(namespace?: string): Promise<EmbeddingRecord[]> {
      const all = Array.from(embeddingRecords.values())
      return namespace === undefined ? all : all.filter((r) => r.namespace === namespace)
    },

    async searchSimilar(
      query: number[],
      threshold: number,
      namespace?: string
    ): Promise<{ record: EmbeddingRecord; similarity: number } | null> {
      const nsIdx = nsIndices.get(nsKey(namespace))
      if (!nsIdx || nsIdx.index.getCurrentCount() === 0) return null

      const { neighbors, distances } = nsIdx.index.searchKnn(query, 1)

      const label = neighbors[0]
      const distance = distances[0]
      if (label === undefined || distance === undefined) return null

      // hnswlib cosine space: distance = 1 − cosine_similarity
      const similarity = 1 - distance
      if (similarity < threshold) return null

      const key = nsIdx.labelToKey.get(label)
      if (!key) return null

      const entry = entries.get(key)
      if (!entry) return null

      // Lazily evict expired entries discovered during search.
      if (isExpired(entry)) {
        await self.delete(key)
        return null
      }

      const record = embeddingRecords.get(key)
      if (!record) return null

      return { record, similarity }
    },
  }

  return self
}
