import type { StoreAdapter, CacheEntry, EmbeddingRecord } from '../types'
import { assertCacheEntry, assertEmbeddingRecord } from '../utils/validate'

// Minimal interface to avoid a hard compile-time dep on better-sqlite3.
interface BetterSqliteDb {
  prepare(sql: string): {
    run(...params: unknown[]): void
    get(...params: unknown[]): unknown
    all(...params: unknown[]): unknown[]
  }
  exec(sql: string): void
}

function initSchema(db: BetterSqliteDb): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS cache_entries (
      key TEXT PRIMARY KEY,
      entry TEXT NOT NULL,
      expires_at INTEGER
    );

    CREATE TABLE IF NOT EXISTS cache_embeddings (
      key TEXT PRIMARY KEY,
      namespace TEXT,
      embedding TEXT NOT NULL,
      created_at INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_embeddings_namespace
      ON cache_embeddings (namespace);

    CREATE INDEX IF NOT EXISTS idx_entries_expires_at
      ON cache_entries (expires_at);
  `)
}

export function sqliteStore(db: unknown): StoreAdapter {
  const sqlite = db as BetterSqliteDb
  initSchema(sqlite)

  const stmtGet = sqlite.prepare('SELECT entry, expires_at FROM cache_entries WHERE key = ?')
  const stmtSet = sqlite.prepare(
    'INSERT OR REPLACE INTO cache_entries (key, entry, expires_at) VALUES (?, ?, ?)'
  )
  const stmtDelete = sqlite.prepare('DELETE FROM cache_entries WHERE key = ?')
  const stmtSetEmbedding = sqlite.prepare(
    'INSERT OR REPLACE INTO cache_embeddings (key, namespace, embedding, created_at) VALUES (?, ?, ?, ?)'
  )
  const stmtDeleteEmbedding = sqlite.prepare('DELETE FROM cache_embeddings WHERE key = ?')
  const stmtListAll = sqlite.prepare('SELECT key, namespace, embedding, created_at FROM cache_embeddings')
  const stmtListByNs = sqlite.prepare(
    'SELECT key, namespace, embedding, created_at FROM cache_embeddings WHERE namespace = ?'
  )

  return {
    async get(key: string): Promise<CacheEntry | null> {
      const row = stmtGet.get(key) as { entry: string; expires_at: number | null } | undefined
      if (!row) return null
      if (row.expires_at !== null && Date.now() > row.expires_at) {
        stmtDelete.run(key)
        stmtDeleteEmbedding.run(key)
        return null
      }
      return assertCacheEntry(JSON.parse(row.entry), 'sqlite')
    },

    async set(key: string, entry: CacheEntry, ttlSeconds?: number): Promise<void> {
      const expiresAt =
        ttlSeconds !== undefined && ttlSeconds > 0 ? Date.now() + ttlSeconds * 1000 : null
      stmtSet.run(key, JSON.stringify(entry), expiresAt)
      stmtSetEmbedding.run(
        key,
        entry.namespace ?? null,
        JSON.stringify(entry.embedding),
        entry.createdAt
      )
    },

    async delete(key: string): Promise<void> {
      stmtDelete.run(key)
      stmtDeleteEmbedding.run(key)
    },

    async listEmbeddings(namespace?: string): Promise<EmbeddingRecord[]> {
      const rows =
        namespace !== undefined
          ? (stmtListByNs.all(namespace) as Array<{
              key: string
              namespace: string | null
              embedding: string
              created_at: number
            }>)
          : (stmtListAll.all() as Array<{
              key: string
              namespace: string | null
              embedding: string
              created_at: number
            }>)

      return rows.map((row) => {
        const parsed = assertEmbeddingRecord(
          {
            key: row.key,
            embedding: JSON.parse(row.embedding) as unknown,
            createdAt: row.created_at,
            ...(row.namespace !== null ? { namespace: row.namespace } : {}),
          },
          'sqlite'
        )
        return parsed
      })
    },
  }
}
