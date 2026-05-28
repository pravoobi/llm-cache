import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { memoryStore } from '../../src/stores/memory'
import type { CacheEntry } from '../../src/types'

function makeEntry(overrides: Partial<CacheEntry> = {}): CacheEntry {
  return {
    prompt: 'test prompt',
    response: { text: 'test response' },
    embedding: [1, 2, 3],
    createdAt: Date.now(),
    ...overrides,
  }
}

describe('memoryStore', () => {
  it('set and get round-trips an entry', async () => {
    const store = memoryStore()
    const entry = makeEntry()
    await store.set('key1', entry)
    const result = await store.get('key1')
    expect(result).toEqual(entry)
  })

  it('get returns null for a missing key', async () => {
    const store = memoryStore()
    expect(await store.get('nonexistent')).toBeNull()
  })

  it('delete removes an entry', async () => {
    const store = memoryStore()
    const entry = makeEntry()
    await store.set('key1', entry)
    await store.delete('key1')
    expect(await store.get('key1')).toBeNull()
  })

  it('delete removes the corresponding embedding record', async () => {
    const store = memoryStore()
    const entry = makeEntry()
    await store.set('key1', entry)
    await store.delete('key1')
    const records = await store.listEmbeddings()
    expect(records.find((r) => r.key === 'key1')).toBeUndefined()
  })

  describe('TTL expiry', () => {
    beforeEach(() => {
      vi.useFakeTimers()
    })
    afterEach(() => {
      vi.useRealTimers()
    })

    it('returns null after TTL expires', async () => {
      const store = memoryStore()
      const now = Date.now()
      const entry = makeEntry({ expiresAt: now + 1000 }) // 1 second TTL
      await store.set('key1', entry, 1)

      vi.setSystemTime(now + 2000) // advance 2 seconds
      expect(await store.get('key1')).toBeNull()
    })

    it('returns entry before TTL expires', async () => {
      const store = memoryStore()
      const now = Date.now()
      const entry = makeEntry({ expiresAt: now + 5000 }) // 5 second TTL
      await store.set('key1', entry, 5)

      vi.setSystemTime(now + 3000) // advance only 3 seconds
      expect(await store.get('key1')).not.toBeNull()
    })
  })

  describe('listEmbeddings', () => {
    it('returns all records when no namespace filter is given', async () => {
      const store = memoryStore()
      await store.set('k1', makeEntry({ namespace: 'a' }))
      await store.set('k2', makeEntry({ namespace: 'b' }))
      const records = await store.listEmbeddings()
      expect(records).toHaveLength(2)
    })

    it('filters by namespace when provided', async () => {
      const store = memoryStore()
      await store.set('k1', makeEntry({ namespace: 'ns-a' }))
      await store.set('k2', makeEntry({ namespace: 'ns-b' }))
      await store.set('k3', makeEntry({ namespace: 'ns-a' }))

      const records = await store.listEmbeddings('ns-a')
      expect(records).toHaveLength(2)
      expect(records.every((r) => r.namespace === 'ns-a')).toBe(true)
    })

    it('returns empty array when no entries match namespace', async () => {
      const store = memoryStore()
      await store.set('k1', makeEntry({ namespace: 'other' }))
      expect(await store.listEmbeddings('missing-ns')).toHaveLength(0)
    })

    it('replaces existing embedding record on re-set', async () => {
      const store = memoryStore()
      await store.set('k1', makeEntry({ embedding: [1, 0, 0] }))
      await store.set('k1', makeEntry({ embedding: [0, 1, 0] }))
      const records = await store.listEmbeddings()
      expect(records).toHaveLength(1)
      expect(records[0]?.embedding).toEqual([0, 1, 0])
    })
  })
})
