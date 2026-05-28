import { describe, it, expect, beforeEach } from 'vitest'
import { hnswMemoryStore, type HnswLib } from '../../src/stores/hnsw-memory'
import type { CacheEntry } from '../../src/types'

// In-process mock of HierarchicalNSW that uses a plain linear cosine scan.
// This lets tests run without the native hnswlib-node binary installed.
class MockHNSW {
  private points = new Map<number, number[]>()
  private deleted = new Set<number>()
  private capacity = 0

  initIndex(maxElements: number) { this.capacity = maxElements }

  addPoint(point: number[], label: number) { this.points.set(label, point) }

  markDelete(label: number) {
    if (!this.points.has(label)) throw new Error('label not found')
    this.deleted.add(label)
  }

  getCurrentCount() { return this.points.size - this.deleted.size }
  getMaxElements() { return this.capacity }
  resizeIndex(n: number) { this.capacity = n }

  searchKnn(query: number[], k: number): { neighbors: number[]; distances: number[] } {
    const candidates = Array.from(this.points.entries()).filter(([l]) => !this.deleted.has(l))
    if (candidates.length === 0) return { neighbors: [], distances: [] }

    const scored = candidates.map(([label, vec]) => {
      let dot = 0, magQ = 0, magV = 0
      for (let i = 0; i < query.length; i++) {
        dot += (query[i] ?? 0) * (vec[i] ?? 0)
        magQ += (query[i] ?? 0) ** 2
        magV += (vec[i] ?? 0) ** 2
      }
      const denom = Math.sqrt(magQ) * Math.sqrt(magV)
      const sim = denom === 0 ? 0 : dot / denom
      return { label, distance: 1 - sim }  // cosine distance
    }).sort((a, b) => a.distance - b.distance)

    return {
      neighbors: scored.slice(0, k).map(s => s.label),
      distances: scored.slice(0, k).map(s => s.distance),
    }
  }
}

const mockLib: HnswLib = { HierarchicalNSW: MockHNSW as never }

function makeEntry(overrides: Partial<CacheEntry> = {}): CacheEntry {
  return {
    prompt: 'test prompt',
    response: 'test response',
    embedding: [1, 0, 0],
    createdAt: Date.now(),
    ...overrides,
  }
}

describe('hnswMemoryStore', () => {
  let store: ReturnType<typeof hnswMemoryStore>

  beforeEach(() => {
    store = hnswMemoryStore(mockLib)
  })

  it('get returns null for missing key', async () => {
    expect(await store.get('missing')).toBeNull()
  })

  it('set/get round-trip', async () => {
    const entry = makeEntry()
    await store.set('k1', entry)
    const result = await store.get('k1')
    expect(result).not.toBeNull()
    expect(result!.prompt).toBe(entry.prompt)
  })

  it('delete removes entry', async () => {
    await store.set('k1', makeEntry())
    await store.delete('k1')
    expect(await store.get('k1')).toBeNull()
  })

  it('listEmbeddings returns all when no namespace', async () => {
    await store.set('k1', makeEntry({ embedding: [1, 0, 0] }))
    await store.set('k2', makeEntry({ embedding: [0, 1, 0] }))
    const records = await store.listEmbeddings()
    expect(records).toHaveLength(2)
  })

  it('listEmbeddings filters by namespace', async () => {
    await store.set('k1', makeEntry({ namespace: 'a', embedding: [1, 0, 0] }))
    await store.set('k2', makeEntry({ namespace: 'b', embedding: [0, 1, 0] }))
    expect(await store.listEmbeddings('a')).toHaveLength(1)
    expect(await store.listEmbeddings('b')).toHaveLength(1)
  })

  it('searchSimilar returns null when store is empty', async () => {
    expect(await store.searchSimilar!([1, 0, 0], 0.5)).toBeNull()
  })

  it('searchSimilar returns exact match for identical vector', async () => {
    await store.set('k1', makeEntry({ embedding: [1, 0, 0] }))
    const result = await store.searchSimilar!([1, 0, 0], 0.9)
    expect(result).not.toBeNull()
    expect(result!.record.key).toBe('k1')
    expect(result!.similarity).toBeCloseTo(1.0)
  })

  it('searchSimilar returns null when best match is below threshold', async () => {
    await store.set('k1', makeEntry({ embedding: [1, 0, 0] }))
    // Orthogonal vector — similarity 0
    expect(await store.searchSimilar!([0, 1, 0], 0.5)).toBeNull()
  })

  it('searchSimilar finds best match among multiple entries', async () => {
    await store.set('k1', makeEntry({ embedding: [1, 0, 0] }))
    await store.set('k2', makeEntry({ embedding: [0.9, 0.1, 0] }))
    await store.set('k3', makeEntry({ embedding: [0, 0, 1] }))
    const result = await store.searchSimilar!([1, 0, 0], 0.5)
    expect(result!.record.key).toBe('k1')
  })

  it('searchSimilar does not cross namespace boundaries', async () => {
    await store.set('k1', makeEntry({ namespace: 'ns-a', embedding: [1, 0, 0] }))
    // Search in ns-b — should return null even though k1 is a perfect match
    expect(await store.searchSimilar!([1, 0, 0], 0.5, 'ns-b')).toBeNull()
  })

  it('searchSimilar respects namespace isolation', async () => {
    await store.set('k1', makeEntry({ namespace: 'ns-a', embedding: [1, 0, 0] }))
    await store.set('k2', makeEntry({ namespace: 'ns-b', embedding: [0, 1, 0] }))
    const resultA = await store.searchSimilar!([1, 0, 0], 0.5, 'ns-a')
    const resultB = await store.searchSimilar!([0, 1, 0], 0.5, 'ns-b')
    expect(resultA!.record.key).toBe('k1')
    expect(resultB!.record.key).toBe('k2')
  })

  it('searchSimilar evicts and returns null for expired entry', async () => {
    const entry = makeEntry({ embedding: [1, 0, 0], expiresAt: Date.now() - 1 })
    await store.set('k1', entry)
    expect(await store.searchSimilar!([1, 0, 0], 0.5)).toBeNull()
    expect(await store.get('k1')).toBeNull()
  })

  it('updating a key replaces its vector in the index', async () => {
    await store.set('k1', makeEntry({ embedding: [1, 0, 0] }))
    // Replace with an orthogonal vector
    await store.set('k1', makeEntry({ embedding: [0, 1, 0] }))
    // Query for the old vector — should no longer match
    const result = await store.searchSimilar!([1, 0, 0], 0.9)
    expect(result).toBeNull()
    // Query for the new vector — should match
    const result2 = await store.searchSimilar!([0, 1, 0], 0.9)
    expect(result2!.record.key).toBe('k1')
  })

  it('index auto-resizes when capacity is exceeded', async () => {
    // hnswMemoryStore starts at INITIAL_CAPACITY=1024; inserting more should
    // trigger resizeIndex without throwing
    const entries = Array.from({ length: 1030 }, (_, i) => ({
      key: `k${i}`,
      entry: makeEntry({ embedding: [Math.random(), Math.random(), Math.random()] }),
    }))
    for (const { key, entry } of entries) {
      await store.set(key, entry)
    }
    const records = await store.listEmbeddings()
    expect(records).toHaveLength(1030)
  })

  it('throws helpful error when hnswlib-node is not installed', async () => {
    // Store without injected lib falls back to dynamic import which will fail
    // in the test environment (hnswlib-node is not a devDep).
    const uninstrumentedStore = hnswMemoryStore()
    await expect(
      uninstrumentedStore.set('k1', makeEntry())
    ).rejects.toThrow('hnswlib-node')
  })
})
