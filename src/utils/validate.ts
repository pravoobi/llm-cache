import type { CacheEntry, EmbeddingRecord } from '../types'

// Minimal structural checks before trusting data deserialized from an external store.
// Defends against a compromised or manually-edited store returning unexpected shapes.

export function assertCacheEntry(val: unknown, source: string): CacheEntry {
  if (
    typeof val !== 'object' ||
    val === null ||
    typeof (val as Record<string, unknown>)['prompt'] !== 'string' ||
    !Array.isArray((val as Record<string, unknown>)['embedding']) ||
    typeof (val as Record<string, unknown>)['createdAt'] !== 'number'
  ) {
    throw new Error(`[llm-cache] Invalid cache entry shape from ${source}`)
  }
  return val as CacheEntry
}

export function assertEmbeddingRecord(val: unknown, source: string): EmbeddingRecord {
  if (
    typeof val !== 'object' ||
    val === null ||
    typeof (val as Record<string, unknown>)['key'] !== 'string' ||
    !Array.isArray((val as Record<string, unknown>)['embedding']) ||
    typeof (val as Record<string, unknown>)['createdAt'] !== 'number'
  ) {
    throw new Error(`[llm-cache] Invalid embedding record shape from ${source}`)
  }
  return val as EmbeddingRecord
}
