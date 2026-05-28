import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { sqliteStore } from '../../src/stores/sqlite'
import type { CacheEntry } from '../../src/types'

function makeEntry(overrides: Partial<CacheEntry> = {}): CacheEntry {
  return {
    prompt: 'hello world',
    response: { text: 'test response' },
    embedding: [0.1, 0.2, 0.3],
    createdAt: Date.now(),
    ...overrides,
  }
}

type EntryRow = { entry: string; expires_at: number | null }
type EmbeddingRow = { key: string; namespace: string | null; embedding: string; created_at: number }

// In-memory mock that mirrors the SQLite schema without requiring better-sqlite3.
function makeMockDb() {
  const entries = new Map<string, EntryRow>()
  const embeddings = new Map<string, EmbeddingRow>()

  const prepare = vi.fn((sql: string) => {
    if (sql.startsWith('SELECT entry')) {
      return {
        run: vi.fn(),
        get: (key: string): EntryRow | undefined => entries.get(key),
        all: vi.fn(),
      }
    }
    if (sql.startsWith('INSERT OR REPLACE INTO cache_entries')) {
      return {
        run: (key: string, entry: string, expiresAt: number | null) => {
          entries.set(key, { entry, expires_at: expiresAt })
        },
        get: vi.fn(),
        all: vi.fn(),
      }
    }
    if (sql.startsWith('DELETE FROM cache_entries')) {
      return {
        run: (key: string) => { entries.delete(key) },
        get: vi.fn(),
        all: vi.fn(),
      }
    }
    if (sql.startsWith('INSERT OR REPLACE INTO cache_embeddings')) {
      return {
        run: (key: string, namespace: string | null, embedding: string, created_at: number) => {
          embeddings.set(key, { key, namespace, embedding, created_at })
        },
        get: vi.fn(),
        all: vi.fn(),
      }
    }
    if (sql.startsWith('DELETE FROM cache_embeddings')) {
      return {
        run: (key: string) => { embeddings.delete(key) },
        get: vi.fn(),
        all: vi.fn(),
      }
    }
    if (sql.includes('cache_embeddings') && !sql.includes('WHERE')) {
      return {
        run: vi.fn(),
        get: vi.fn(),
        all: (): EmbeddingRow[] => Array.from(embeddings.values()),
      }
    }
    if (sql.includes('WHERE namespace = ?')) {
      return {
        run: vi.fn(),
        get: vi.fn(),
        all: (...args: unknown[]): EmbeddingRow[] =>
          Array.from(embeddings.values()).filter((r) => r.namespace === (args[0] as string)),
      }
    }
    return { run: vi.fn(), get: vi.fn(), all: vi.fn(() => []) }
  })

  return {
    exec: vi.fn(),
    prepare,
    _entries: entries,
    _embeddings: embeddings,
  }
}

describe('sqliteStore', () => {
  let db: ReturnType<typeof makeMockDb>
  let store: ReturnType<typeof sqliteStore>

  beforeEach(() => {
    db = makeMockDb()
    store = sqliteStore(db)
  })

  it('initialises the schema on creation', () => {
    expect(db.exec).toHaveBeenCalledOnce()
  })

  it('get returns null for a missing key', async () => {
    expect(await store.get('nonexistent')).toBeNull()
  })

  it('set and get round-trips an entry', async () => {
    const entry = makeEntry()
    await store.set('key1', entry)
    const result = await store.get('key1')
    expect(result).not.toBeNull()
    expect(result!.prompt).toBe(entry.prompt)
    expect(result!.response).toEqual(entry.response)
    expect(result!.embedding).toEqual(entry.embedding)
  })

  it('set stores an embedding record alongside the cache entry', async () => {
    const entry = makeEntry()
    await store.set('key1', entry)
    const records = await store.listEmbeddings()
    expect(records).toHaveLength(1)
    expect(records[0]!.key).toBe('key1')
    expect(records[0]!.embedding).toEqual(entry.embedding)
  })

  it('delete removes entry and embedding record', async () => {
    await store.set('key1', makeEntry())
    await store.delete('key1')
    expect(await store.get('key1')).toBeNull()
    expect(await store.listEmbeddings()).toHaveLength(0)
  })

  describe('TTL expiry', () => {
    beforeEach(() => { vi.useFakeTimers() })
    afterEach(() => { vi.useRealTimers() })

    it('returns null after TTL expires', async () => {
      const now = Date.now()
      await store.set('key1', makeEntry(), 1) // 1 second TTL

      vi.setSystemTime(now + 2000)
      expect(await store.get('key1')).toBeNull()
    })

    it('returns entry before TTL expires', async () => {
      const now = Date.now()
      await store.set('key1', makeEntry(), 5) // 5 second TTL

      vi.setSystemTime(now + 3000)
      expect(await store.get('key1')).not.toBeNull()
    })

    it('evicts the embedding record when an expired entry is fetched', async () => {
      const now = Date.now()
      await store.set('key1', makeEntry(), 1)
      vi.setSystemTime(now + 2000)

      await store.get('key1') // triggers eviction
      expect(await store.listEmbeddings()).toHaveLength(0)
    })
  })

  describe('listEmbeddings', () => {
    it('returns all records when no namespace filter is given', async () => {
      await store.set('k1', makeEntry({ namespace: 'a' }))
      await store.set('k2', makeEntry({ namespace: 'b' }))
      expect(await store.listEmbeddings()).toHaveLength(2)
    })

    it('filters by namespace when provided', async () => {
      await store.set('k1', makeEntry({ namespace: 'ns-a' }))
      await store.set('k2', makeEntry({ namespace: 'ns-b' }))
      await store.set('k3', makeEntry({ namespace: 'ns-a' }))

      const records = await store.listEmbeddings('ns-a')
      expect(records).toHaveLength(2)
      expect(records.every((r) => r.namespace === 'ns-a')).toBe(true)
    })

    it('returns empty array when no entries match namespace', async () => {
      await store.set('k1', makeEntry({ namespace: 'other' }))
      expect(await store.listEmbeddings('missing-ns')).toHaveLength(0)
    })

    it('replaces existing embedding record on re-set', async () => {
      await store.set('k1', makeEntry({ embedding: [1, 0, 0] }))
      await store.set('k1', makeEntry({ embedding: [0, 1, 0] }))
      const records = await store.listEmbeddings()
      expect(records).toHaveLength(1)
      expect(records[0]!.embedding).toEqual([0, 1, 0])
    })
  })

  describe('input validation', () => {
    it('throws on a malformed cache entry from the store', async () => {
      db._entries.set('badkey', { entry: JSON.stringify({ not: 'valid' }), expires_at: null })
      await expect(store.get('badkey')).rejects.toThrow('Invalid cache entry shape')
    })

    it('throws on a malformed embedding record from the store', async () => {
      // Poison the embeddings table with a non-array embedding value
      db._embeddings.set('badkey', {
        key: 'badkey',
        namespace: null,
        embedding: '"not-an-array"',
        created_at: Date.now(),
      })
      await expect(store.listEmbeddings()).rejects.toThrow('Invalid embedding record shape')
    })
  })
})
