import type { EmbeddingRecord } from './types'

// Cosine similarity in [-1, 1]; returns 0 for zero-magnitude vectors to avoid
// division-by-zero without masking legitimate near-zero similarities.
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(
      `Embedding dimension mismatch: vector a has ${a.length} dimensions, vector b has ${b.length} dimensions`
    )
  }

  let dot = 0
  let magA = 0
  let magB = 0

  for (let i = 0; i < a.length; i++) {
    const ai = a[i] ?? 0
    const bi = b[i] ?? 0
    dot += ai * bi
    magA += ai * ai
    magB += bi * bi
  }

  const denom = Math.sqrt(magA) * Math.sqrt(magB)
  if (denom === 0) return 0

  return dot / denom
}

export function findBestMatch(
  query: number[],
  records: EmbeddingRecord[],
  threshold: number
): { record: EmbeddingRecord; similarity: number } | null {
  // Linear scan becomes costly past ~10k vectors. Switch to hnswMemoryStore()
  // for in-process ANN search, or pgvector for multi-process deployments.
  if (records.length > 10_000) {
    console.warn(
      `[llm-cache] Scanning ${records.length} embeddings with O(n) linear search. ` +
        'Use hnswMemoryStore() for fast in-process ANN, or pgvector for multi-process deployments.'
    )
  }

  let bestSimilarity = -Infinity
  let bestRecord: EmbeddingRecord | null = null

  for (const record of records) {
    const similarity = cosineSimilarity(query, record.embedding)
    if (similarity > bestSimilarity) {
      bestSimilarity = similarity
      bestRecord = record
    }
  }

  if (bestRecord === null || bestSimilarity < threshold) return null

  return { record: bestRecord, similarity: bestSimilarity }
}
