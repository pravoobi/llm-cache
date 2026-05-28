import type { StoreAdapter, CacheEntry, EmbeddingRecord } from '../types'

// Minimal pg Pool interface to avoid a hard compile-time dep on pg.
interface PgPool {
  query(sql: string, params?: unknown[]): Promise<{ rows: unknown[] }>
  end(): Promise<void>
}

// Row shape returned by the llm_cache table queries.
interface LlmCacheRow {
  key: string
  prompt: string
  response: unknown
  embedding: string // pg returns vector as comma-separated string inside brackets
  namespace: string | null
  created_at: Date
  expires_at: Date | null
}

// pgvector default dimension — callers using non-1536-dim models should adjust
// the table DDL via a migration before using this store.
const VECTOR_DIM = 1536

async function initSchema(pool: PgPool): Promise<void> {
  // pgvector extension must be installed by a superuser before this runs.
  await pool.query('CREATE EXTENSION IF NOT EXISTS vector')
  await pool.query(`
    CREATE TABLE IF NOT EXISTS llm_cache (
      key TEXT PRIMARY KEY,
      prompt TEXT NOT NULL,
      response JSONB,
      embedding vector(${VECTOR_DIM}),
      namespace TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      expires_at TIMESTAMPTZ
    )
  `)
  // IVFFlat index requires at least 1 row before ANALYZE is meaningful,
  // so we create it without specifying lists — it will be rebuilt on VACUUM ANALYZE.
  await pool.query(`
    CREATE INDEX IF NOT EXISTS idx_llm_cache_embedding
      ON llm_cache USING ivfflat (embedding vector_cosine_ops)
  `)
  await pool.query('CREATE INDEX IF NOT EXISTS idx_llm_cache_namespace ON llm_cache (namespace)')
  await pool.query('CREATE INDEX IF NOT EXISTS idx_llm_cache_expires_at ON llm_cache (expires_at)')
}

function parseEmbedding(raw: string): number[] {
  // pgvector returns embeddings as "[1.0,2.0,...]" strings.
  return raw
    .replace(/^\[/, '')
    .replace(/\]$/, '')
    .split(',')
    .map(Number)
}

export function pgvectorStore(pool: unknown): StoreAdapter {
  const pg = pool as PgPool
  const ready = initSchema(pg)

  return {
    async get(key: string): Promise<CacheEntry | null> {
      await ready
      const result = await pg.query(
        'SELECT * FROM llm_cache WHERE key = $1',
        [key]
      )
      const row = (result.rows[0] ?? null) as LlmCacheRow | null
      if (!row) return null

      if (row.expires_at !== null && new Date() > row.expires_at) {
        await pg.query('DELETE FROM llm_cache WHERE key = $1', [key])
        return null
      }

      return {
        prompt: row.prompt,
        response: row.response,
        embedding: parseEmbedding(row.embedding),
        createdAt: row.created_at.getTime(),
        ...(row.namespace !== null ? { namespace: row.namespace } : {}),
        ...(row.expires_at !== null ? { expiresAt: row.expires_at.getTime() } : {}),
      }
    },

    async set(key: string, entry: CacheEntry, ttlSeconds?: number): Promise<void> {
      await ready
      const expiresAt =
        ttlSeconds !== undefined && ttlSeconds > 0
          ? new Date(Date.now() + ttlSeconds * 1000)
          : null

      // pgvector expects the embedding as a bracketed comma-separated string.
      const embeddingStr = `[${entry.embedding.join(',')}]`

      await pg.query(
        `INSERT INTO llm_cache (key, prompt, response, embedding, namespace, created_at, expires_at)
         VALUES ($1, $2, $3, $4, $5, $6, $7)
         ON CONFLICT (key) DO UPDATE SET
           prompt = EXCLUDED.prompt,
           response = EXCLUDED.response,
           embedding = EXCLUDED.embedding,
           namespace = EXCLUDED.namespace,
           created_at = EXCLUDED.created_at,
           expires_at = EXCLUDED.expires_at`,
        [
          key,
          entry.prompt,
          JSON.stringify(entry.response),
          embeddingStr,
          entry.namespace ?? null,
          new Date(entry.createdAt),
          expiresAt,
        ]
      )
    },

    async delete(key: string): Promise<void> {
      await ready
      await pg.query('DELETE FROM llm_cache WHERE key = $1', [key])
    },

    async listEmbeddings(namespace?: string): Promise<EmbeddingRecord[]> {
      await ready
      const result =
        namespace !== undefined
          ? await pg.query(
              'SELECT key, namespace, embedding, created_at FROM llm_cache WHERE namespace = $1',
              [namespace]
            )
          : await pg.query('SELECT key, namespace, embedding, created_at FROM llm_cache')

      return (result.rows as Array<{
        key: string
        namespace: string | null
        embedding: string
        created_at: Date
      }>).map((row) => ({
        key: row.key,
        embedding: parseEmbedding(row.embedding),
        createdAt: row.created_at.getTime(),
        ...(row.namespace !== null ? { namespace: row.namespace } : {}),
      }))
    },

    async close(): Promise<void> {
      await pg.end()
    },
  }
}
