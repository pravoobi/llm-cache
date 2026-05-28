import { describe, it, expect, beforeEach } from 'vitest'
import { pgvectorStore } from '../../src/stores/pgvector'
import type { CacheEntry } from '../../src/types'

interface RecordedQuery { sql: string; params?: unknown[] }

function makeMockPool(
  responder?: (sql: string, params?: unknown[]) => { rows: unknown[] }
) {
  const queries: RecordedQuery[] = []
  return {
    async query(sql: string, params?: unknown[]) {
      queries.push({ sql, params })
      return responder ? responder(sql, params) : { rows: [] }
    },
    async end() {},
    _queries: queries,
  }
}

function makeEntry(overrides: Partial<CacheEntry> = {}): CacheEntry {
  return {
    prompt: 'hello',
    response: { text: 'world' },
    embedding: [0.1, 0.2, 0.3],
    createdAt: Date.now(),
    ...overrides,
  }
}

// Wait for initSchema to complete by triggering the first real query.
async function settle(store: ReturnType<typeof pgvectorStore>) {
  await store.get('__settle__')
}

describe('pgvectorStore — dimensions option', () => {
  it('uses 1536 by default', async () => {
    const pool = makeMockPool()
    const store = pgvectorStore(pool)
    await settle(store)
    const ddl = pool._queries.find(q => q.sql.includes('CREATE TABLE'))
    expect(ddl?.sql).toContain('vector(1536)')
  })

  it('uses configured dimension for Cohere (1024)', async () => {
    const pool = makeMockPool()
    const store = pgvectorStore(pool, { dimensions: 1024 })
    await settle(store)
    const ddl = pool._queries.find(q => q.sql.includes('CREATE TABLE'))
    expect(ddl?.sql).toContain('vector(1024)')
    expect(ddl?.sql).not.toContain('vector(1536)')
  })

  it('uses configured dimension for local model (384)', async () => {
    const pool = makeMockPool()
    const store = pgvectorStore(pool, { dimensions: 384 })
    await settle(store)
    const ddl = pool._queries.find(q => q.sql.includes('CREATE TABLE'))
    expect(ddl?.sql).toContain('vector(384)')
  })
})

describe('pgvectorStore — CRUD', () => {
  let pool: ReturnType<typeof makeMockPool>
  let store: ReturnType<typeof pgvectorStore>

  beforeEach(() => {
    pool = makeMockPool()
    store = pgvectorStore(pool)
  })

  it('get returns null for missing key', async () => {
    expect(await store.get('missing')).toBeNull()
  })

  it('get returns null for expired entry', async () => {
    const expiredRow = {
      key: 'k1',
      prompt: 'p',
      response: {},
      embedding: '[0.1,0.2,0.3]',
      namespace: null,
      created_at: new Date(),
      expires_at: new Date(Date.now() - 1000),
    }
    const respondingPool = makeMockPool((sql) =>
      sql.includes('SELECT') ? { rows: [expiredRow] } : { rows: [] }
    )
    const s = pgvectorStore(respondingPool)
    expect(await s.get('k1')).toBeNull()
    const deleted = respondingPool._queries.find(q => q.sql.includes('DELETE'))
    expect(deleted).toBeDefined()
  })

  it('get returns parsed entry for valid row', async () => {
    const now = new Date()
    const row = {
      key: 'k1',
      prompt: 'hello',
      response: { text: 'world' },
      embedding: '[0.1,0.2,0.3]',
      namespace: 'ns1',
      created_at: now,
      expires_at: null,
    }
    const respondingPool = makeMockPool((sql) =>
      sql.includes('SELECT') ? { rows: [row] } : { rows: [] }
    )
    const s = pgvectorStore(respondingPool)
    const entry = await s.get('k1')
    expect(entry).not.toBeNull()
    expect(entry!.prompt).toBe('hello')
    expect(entry!.namespace).toBe('ns1')
    expect(entry!.embedding).toEqual([0.1, 0.2, 0.3])
    expect(entry!.response).toEqual({ text: 'world' })
  })

  it('set calls INSERT with bracketed embedding string', async () => {
    const entry = makeEntry({ embedding: [0.1, 0.2, 0.3] })
    await store.set('k1', entry)
    const insert = pool._queries.find(q => q.sql.includes('INSERT'))
    expect(insert).toBeDefined()
    expect(insert!.params).toContain('[0.1,0.2,0.3]')
  })

  it('set passes TTL as expires_at when provided', async () => {
    const entry = makeEntry()
    const before = Date.now()
    await store.set('k1', entry, 3600)
    const insert = pool._queries.find(q => q.sql.includes('INSERT'))
    const expiresAt = insert!.params![6] as Date
    expect(expiresAt).toBeInstanceOf(Date)
    expect(expiresAt.getTime()).toBeGreaterThan(before + 3590_000)
  })

  it('set passes null expires_at when no TTL', async () => {
    await store.set('k1', makeEntry())
    const insert = pool._queries.find(q => q.sql.includes('INSERT'))
    expect(insert!.params![6]).toBeNull()
  })

  it('delete calls DELETE with key param', async () => {
    await store.delete('k1')
    const del = pool._queries.find(q => q.sql.includes('DELETE'))
    expect(del?.params).toContain('k1')
  })

  it('listEmbeddings without namespace queries all rows', async () => {
    const respondingPool = makeMockPool((sql) =>
      sql.includes('SELECT') ? { rows: [] } : { rows: [] }
    )
    const s = pgvectorStore(respondingPool)
    await s.listEmbeddings()
    const select = respondingPool._queries.find(
      q => q.sql.includes('SELECT') && q.sql.includes('embedding')
    )
    expect(select?.sql).not.toContain('WHERE namespace')
  })

  it('listEmbeddings with namespace adds WHERE clause', async () => {
    const respondingPool = makeMockPool(() => ({ rows: [] }))
    const s = pgvectorStore(respondingPool)
    await s.listEmbeddings('ns-a')
    const select = respondingPool._queries.find(
      q => q.sql.includes('WHERE namespace')
    )
    expect(select?.params).toContain('ns-a')
  })

  it('close calls pool.end', async () => {
    let ended = false
    const p = { ...pool, async end() { ended = true } }
    const s = pgvectorStore(p)
    await s.close?.()
    expect(ended).toBe(true)
  })
})
