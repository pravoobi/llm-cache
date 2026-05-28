import { describe, it, expect, vi, beforeEach } from 'vitest'
import { redisStore } from '../../src/stores/redis'
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

function makeMockClient() {
  const store = new Map<string, string>()
  const hashes = new Map<string, Map<string, string>>()

  return {
    get: vi.fn(async (key: string) => store.get(key) ?? null),
    set: vi.fn(async (key: string, value: string) => { store.set(key, value) }),
    setex: vi.fn(async (key: string, _ttl: number, value: string) => { store.set(key, value) }),
    del: vi.fn(async (key: string) => { store.delete(key) }),
    hset: vi.fn(async (hash: string, field: string, value: string) => {
      if (!hashes.has(hash)) hashes.set(hash, new Map())
      hashes.get(hash)!.set(field, value)
    }),
    hdel: vi.fn(async (hash: string, field: string) => {
      hashes.get(hash)?.delete(field)
    }),
    hgetall: vi.fn(async (hash: string) => {
      const h = hashes.get(hash)
      if (!h || h.size === 0) return null
      return Object.fromEntries(h)
    }),
    quit: vi.fn(async () => {}),
    _store: store,
    _hashes: hashes,
  }
}

describe('redisStore', () => {
  let client: ReturnType<typeof makeMockClient>
  let store: ReturnType<typeof redisStore>

  beforeEach(() => {
    client = makeMockClient()
    store = redisStore(client)
  })

  it('returns null for missing key', async () => {
    expect(await store.get('missing')).toBeNull()
  })

  it('set/get round-trip', async () => {
    const entry = makeEntry()
    await store.set('key1', entry)
    const result = await store.get('key1')
    expect(result).not.toBeNull()
    expect(result!.prompt).toBe(entry.prompt)
    expect(result!.response).toEqual(entry.response)
  })

  it('set with TTL calls setex', async () => {
    const entry = makeEntry()
    await store.set('key1', entry, 3600)
    expect(client.setex).toHaveBeenCalledWith(
      'llm-cache:entry:key1',
      3600,
      expect.any(String)
    )
  })

  it('set without TTL calls set', async () => {
    const entry = makeEntry()
    await store.set('key1', entry)
    expect(client.set).toHaveBeenCalledWith(
      'llm-cache:entry:key1',
      expect.any(String)
    )
  })

  it('delete removes entry and embedding record', async () => {
    const entry = makeEntry({ namespace: 'ns1' })
    await store.set('key1', entry)
    await store.delete('key1')
    expect(await store.get('key1')).toBeNull()
    expect(client.del).toHaveBeenCalledWith('llm-cache:entry:key1')
    expect(client.hdel).toHaveBeenCalled()
  })

  it('listEmbeddings returns all records when no namespace given', async () => {
    await store.set('key1', makeEntry({ embedding: [1, 0, 0] }))
    await store.set('key2', makeEntry({ embedding: [0, 1, 0] }))
    const records = await store.listEmbeddings()
    expect(records).toHaveLength(2)
  })

  it('listEmbeddings filters by namespace', async () => {
    await store.set('key1', makeEntry({ namespace: 'team-a', embedding: [1, 0, 0] }))
    await store.set('key2', makeEntry({ namespace: 'team-b', embedding: [0, 1, 0] }))

    // namespace values are base64url-encoded in the Redis hash key
    const teamA = await store.listEmbeddings('team-a')
    const teamB = await store.listEmbeddings('team-b')

    expect(teamA).toHaveLength(1)
    expect(teamB).toHaveLength(1)
  })

  it('namespace values with colons are encoded safely', async () => {
    // A namespace containing a colon must not split the Redis key hierarchy
    const entry = makeEntry({ namespace: 'tenant:admin' })
    await store.set('key1', entry)
    const hashKeyArg = client.hset.mock.calls[0]?.[0] as string
    expect(hashKeyArg).not.toContain('tenant:admin')
    expect(hashKeyArg).toMatch(/^llm-cache:embeddings:/)
  })

  it('get returns null for invalid stored shape and throws', async () => {
    // Directly poison the Redis store with a malformed entry
    client._store.set('llm-cache:entry:badkey', JSON.stringify({ not: 'a valid entry' }))
    await expect(store.get('badkey')).rejects.toThrow('Invalid cache entry shape')
  })

  it('close calls client.quit', async () => {
    await store.close?.()
    expect(client.quit).toHaveBeenCalled()
  })
})
