import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createCache } from '../src/cache'
import { memoryStore } from '../src/stores/memory'
import type { StoreAdapter, CacheEntry, EmbeddingRecord, LLMCacheConfig } from '../src/types'

// Deterministic embedder that returns a fixed vector for any input,
// allowing us to control similarity by swapping vectors in tests.
function fixedEmbedder(vec: number[]) {
  return vi.fn(async (_text: string) => vec)
}

// Creates a semantically-controlled embedder: maps specific prompt prefixes
// to specific vectors so we can craft exact similarity scenarios.
function mappedEmbedder(map: Record<string, number[]>, fallback: number[]) {
  return vi.fn(async (text: string) => {
    for (const [prefix, vec] of Object.entries(map)) {
      if (text.startsWith(prefix)) return vec
    }
    return fallback
  })
}

describe('createCache', () => {
  // --- Test 1: exact hit ---
  it('returns cached response on exact prompt match without calling fn', async () => {
    const embedder = fixedEmbedder([1, 0, 0])
    const cache = createCache({ embedder, store: 'memory', threshold: 0.9 })

    const fn = vi.fn(async () => 'llm-response')
    await cache.wrap('hello', fn)           // miss — populates cache
    const result = await cache.wrap('hello', fn) // exact hit

    expect(result.hit).toBe(true)
    expect(result.layer).toBe('exact')
    expect(result.value).toBe('llm-response')
    expect(fn).toHaveBeenCalledTimes(1) // fn only called on first miss
  })

  // --- Test 2: semantic hit ---
  it('returns cached response for semantically similar prompt above threshold', async () => {
    // v1 and v2 are very close (cos sim ≈ 0.999), above threshold 0.9
    const v1 = [1, 0.01, 0]
    const v2 = [1, 0.02, 0]

    const embedder = mappedEmbedder({ 'store this': v1, 'similar to': v2 }, [0, 0, 1])
    const cache = createCache({ embedder, store: 'memory', threshold: 0.9 })

    const fn = vi.fn(async () => 'original-response')
    await cache.wrap('store this prompt', fn)

    const fn2 = vi.fn(async () => 'should-not-be-called')
    const result = await cache.wrap('similar to store this prompt', fn2)

    expect(result.hit).toBe(true)
    expect(result.layer).toBe('semantic')
    expect(result.similarity).toBeGreaterThan(0.9)
    expect(fn2).not.toHaveBeenCalled()
  })

  // --- Test 3: cache miss ---
  it('calls fn and stores entry when prompt is below threshold', async () => {
    // orthogonal vectors → similarity = 0
    const embedder = mappedEmbedder({ 'prompt-a': [1, 0, 0], 'prompt-b': [0, 1, 0] }, [0, 0, 1])
    const cache = createCache({ embedder, store: 'memory', threshold: 0.9 })

    const fn1 = vi.fn(async () => 'response-a')
    await cache.wrap('prompt-a first', fn1)

    const fn2 = vi.fn(async () => 'response-b')
    const result = await cache.wrap('prompt-b second', fn2)

    expect(result.hit).toBe(false)
    expect(result.layer).toBe('miss')
    expect(fn2).toHaveBeenCalledTimes(1)
    expect(result.value).toBe('response-b')
  })

  // --- Test 4: bypass ---
  it('always calls fn when bypass is true, never reads or writes cache', async () => {
    const embedder = fixedEmbedder([1, 0, 0])
    const store = memoryStore()
    const setSpy = vi.spyOn(store, 'set')
    const getSpy = vi.spyOn(store, 'get')

    const cache = createCache({ embedder, store, threshold: 0.9 })
    const fn = vi.fn(async () => 'bypass-response')

    const result = await cache.wrap('hello', fn, { bypass: true })

    expect(result.hit).toBe(false)
    expect(result.layer).toBe('miss')
    expect(fn).toHaveBeenCalledTimes(1)
    expect(getSpy).not.toHaveBeenCalled()
    expect(setSpy).not.toHaveBeenCalled()
  })

  // --- Test 5: namespace isolation ---
  it('does not return cross-namespace hits', async () => {
    const embedder = fixedEmbedder([1, 0, 0]) // identical vector for both
    const cache = createCache({ embedder, store: 'memory', threshold: 0.9 })

    const fn1 = vi.fn(async () => 'ns-a-response')
    await cache.wrap('same prompt', fn1, { namespace: 'ns-a' })

    const fn2 = vi.fn(async () => 'ns-b-response')
    const result = await cache.wrap('same prompt', fn2, { namespace: 'ns-b' })

    // Different namespace → miss, fn2 must be called
    expect(fn2).toHaveBeenCalledTimes(1)
    expect(result.value).toBe('ns-b-response')
  })

  // --- Test 6: context isolation ---
  it('does not return cross-context hits', async () => {
    const embedder = fixedEmbedder([1, 0, 0])
    const cache = createCache({ embedder, store: 'memory', threshold: 0.9 })

    const fn1 = vi.fn(async () => 'ctx-a-response')
    await cache.wrap('same prompt', fn1, { context: 'context-a' })

    const fn2 = vi.fn(async () => 'ctx-b-response')
    const result = await cache.wrap('same prompt', fn2, { context: 'context-b' })

    expect(fn2).toHaveBeenCalledTimes(1)
    expect(result.value).toBe('ctx-b-response')
  })

  // --- Test 7: TTL expiry ---
  it('treats expired entry as a miss', async () => {
    vi.useFakeTimers()

    try {
      const now = Date.now()
      const embedder = fixedEmbedder([1, 0, 0])

      // Build a store that returns an already-expired entry on first get.
      const expiredEntry: CacheEntry = {
        prompt: 'hello',
        response: 'stale',
        embedding: [1, 0, 0],
        createdAt: now - 10_000,
        expiresAt: now - 1, // expired 1 ms ago
      }

      const mockStore: StoreAdapter = {
        get: vi.fn(async (_key: string) => expiredEntry),
        set: vi.fn(async () => {}),
        delete: vi.fn(async () => {}),
        listEmbeddings: vi.fn(async () => [] as EmbeddingRecord[]),
      }

      const cache = createCache({ embedder, store: mockStore, threshold: 0.9 })
      const fn = vi.fn(async () => 'fresh-response')

      // The memory store impl evicts on get() — but our mock just returns the
      // expired entry. The cache.ts layer itself must NOT treat an expiredAt entry
      // returned by the store as a hit (store handles eviction, but we test that
      // a freshly-missed cache path calls fn).
      // Use real memoryStore to test TTL end-to-end:
      const realStore = memoryStore()
      const cacheReal = createCache({ embedder: fixedEmbedder([1, 0, 0]), store: realStore, threshold: 0.9 })

      const fn2 = vi.fn(async () => 'first')
      await cacheReal.wrap('hello', fn2, { ttl: 1 }) // store with 1s TTL
      expect(fn2).toHaveBeenCalledTimes(1)

      vi.setSystemTime(now + 5000) // advance 5 seconds past TTL

      const fn3 = vi.fn(async () => 'after-expiry')
      const result = await cacheReal.wrap('hello', fn3, { ttl: 1 })
      expect(fn3).toHaveBeenCalledTimes(1) // expired → miss → fn called
      expect(result.value).toBe('after-expiry')
      expect(result.hit).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })

  // --- Test 8: fn() throws → error propagates, nothing cached ---
  it('propagates fn() errors without caching', async () => {
    const embedder = fixedEmbedder([1, 0, 0])
    const store = memoryStore()
    const setSpy = vi.spyOn(store, 'set')

    const cache = createCache({ embedder, store, threshold: 0.9 })
    const fn = vi.fn(async () => {
      throw new Error('LLM API failure')
    })

    await expect(cache.wrap('hello', fn)).rejects.toThrow('LLM API failure')
    expect(setSpy).not.toHaveBeenCalled()
  })

  // --- Test 9: embedder throws → onError called, fn() used as fallback ---
  it('calls onError and falls back to fn() when the embedder throws', async () => {
    const failingEmbedder = vi.fn(async () => {
      throw new Error('embedding service down')
    })

    const onError = vi.fn()
    const cache = createCache({
      embedder: failingEmbedder,
      store: 'memory',
      threshold: 0.9,
      onError,
    })

    const fn = vi.fn(async () => 'fallback-value')
    const result = await cache.wrap('hello', fn)

    expect(onError).toHaveBeenCalledWith(expect.any(Error))
    expect(fn).toHaveBeenCalledTimes(1)
    expect(result.value).toBe('fallback-value')
    expect(result.hit).toBe(false)
    expect(result.layer).toBe('miss')
  })

  describe('stats()', () => {
    it('tracks hit rate correctly', async () => {
      const embedder = fixedEmbedder([1, 0, 0])
      const cache = createCache({ embedder, store: 'memory', threshold: 0.9 })

      await cache.wrap('p1', async () => 'r1') // miss
      await cache.wrap('p1', async () => 'r1') // exact hit

      const s = cache.stats()
      expect(s.hitRate).toBeCloseTo(0.5)
    })

    it('returns 0 hitRate when nothing has been called', () => {
      const cache = createCache({ embedder: fixedEmbedder([1, 0, 0]), store: 'memory' })
      expect(cache.stats().hitRate).toBe(0)
    })
  })

  describe('invalidate()', () => {
    it('removes an entry so the next call is a miss', async () => {
      const embedder = fixedEmbedder([1, 0, 0])
      const cache = createCache({ embedder, store: 'memory', threshold: 0.9 })

      const fn = vi.fn(async () => 'value')
      await cache.wrap('hello', fn) // store it
      await cache.invalidate('hello')

      await cache.wrap('hello', fn) // should miss again
      expect(fn).toHaveBeenCalledTimes(2)
    })
  })

  describe('flush()', () => {
    it('clears all entries', async () => {
      const embedder = fixedEmbedder([1, 0, 0])
      const store = memoryStore()
      const cache = createCache({ embedder, store, threshold: 0.9 })

      await cache.wrap('p1', async () => 'v1')
      await cache.wrap('p2', async () => 'v2')
      await cache.flush()

      const records = await store.listEmbeddings()
      expect(records).toHaveLength(0)
    })
  })

  describe('searchSimilar branch (ANN store)', () => {
    it('calls store.searchSimilar instead of listEmbeddings when available', async () => {
      const embedder = fixedEmbedder([1, 0, 0])
      const base = memoryStore()
      const searchSimilar = vi.fn(async () => null)
      const annStore: StoreAdapter = { ...base, searchSimilar }
      const listSpy = vi.spyOn(annStore, 'listEmbeddings')

      const cache = createCache({ embedder, store: annStore, threshold: 0.9 })
      await cache.wrap('hello', async () => 'v')

      expect(searchSimilar).toHaveBeenCalledOnce()
      expect(listSpy).not.toHaveBeenCalled()
    })

    it('returns semantic hit from searchSimilar result', async () => {
      const embedder = fixedEmbedder([1, 0, 0])
      const base = memoryStore()

      // Prime the underlying store with one entry via a plain cache instance
      await createCache({ embedder, store: base }).wrap('original', async () => 'cached-value')
      const [storedRecord] = await base.listEmbeddings()

      const searchSimilar = vi.fn(async () => ({
        record: storedRecord!,
        similarity: 0.95,
      }))
      const annStore: StoreAdapter = { ...base, searchSimilar }
      const cache = createCache({ embedder, store: annStore, threshold: 0.9 })

      const fn = vi.fn(async () => 'should-not-be-called')
      const result = await cache.wrap('similar prompt', fn)

      expect(fn).not.toHaveBeenCalled()
      expect(result.hit).toBe(true)
      expect(result.layer).toBe('semantic')
      expect(result.similarity).toBe(0.95)
      expect(result.value).toBe('cached-value')
      expect(searchSimilar).toHaveBeenCalledOnce()
    })

    it('falls back to listEmbeddings when searchSimilar throws', async () => {
      const embedder = fixedEmbedder([1, 0, 0])
      const base = memoryStore()
      const onError = vi.fn()
      const searchSimilar = vi.fn(async () => { throw new Error('ANN index unavailable') })
      const annStore: StoreAdapter = { ...base, searchSimilar }

      const cache = createCache({ embedder, store: annStore, threshold: 0.9, onError })
      const fn = vi.fn(async () => 'fallback')
      const result = await cache.wrap('hello', fn)

      expect(onError).toHaveBeenCalledWith(expect.any(Error))
      expect(result.hit).toBe(false)
      expect(result.value).toBe('fallback')
    })
  })
})
