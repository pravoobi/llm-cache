import { describe, it, expect } from 'vitest'
import { cosineSimilarity, findBestMatch } from '../src/similarity'
import type { EmbeddingRecord } from '../src/types'

describe('cosineSimilarity', () => {
  it('returns 1.0 for identical vectors', () => {
    const v = [1, 2, 3]
    expect(cosineSimilarity(v, v)).toBeCloseTo(1.0)
  })

  it('returns 0.0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBeCloseTo(0.0)
  })

  it('returns 0 for a zero vector without throwing', () => {
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0)
    expect(cosineSimilarity([1, 2, 3], [0, 0, 0])).toBe(0)
    expect(cosineSimilarity([0, 0, 0], [0, 0, 0])).toBe(0)
  })

  it('throws a descriptive error for dimension mismatch', () => {
    expect(() => cosineSimilarity([1, 2, 3], [1, 2])).toThrow('dimension mismatch')
    expect(() => cosineSimilarity([1, 2, 3], [1, 2])).toThrow('3')
    expect(() => cosineSimilarity([1, 2, 3], [1, 2])).toThrow('2')
  })

  it('computes correct similarity for known vectors', () => {
    // [1,0] vs [1,1]/sqrt(2) ≈ 0.707
    const sim = cosineSimilarity([1, 0], [1, 1])
    expect(sim).toBeCloseTo(1 / Math.sqrt(2), 5)
  })
})

describe('findBestMatch', () => {
  const makeRecord = (key: string, embedding: number[], namespace?: string): EmbeddingRecord => ({
    key,
    embedding,
    createdAt: Date.now(),
    ...(namespace !== undefined ? { namespace } : {}),
  })

  it('returns null when records is empty', () => {
    expect(findBestMatch([1, 0], [], 0.8)).toBeNull()
  })

  it('returns null when all similarities are below threshold', () => {
    const records = [makeRecord('a', [0, 1]), makeRecord('b', [0, 1])]
    // [1,0] vs [0,1] = 0 similarity, threshold 0.5
    expect(findBestMatch([1, 0], records, 0.5)).toBeNull()
  })

  it('returns the best match above threshold', () => {
    const records = [
      makeRecord('low', [0, 1]),   // orthogonal to query
      makeRecord('high', [1, 0.1]), // nearly parallel to query
    ]
    const result = findBestMatch([1, 0], records, 0.5)
    expect(result).not.toBeNull()
    expect(result!.record.key).toBe('high')
    expect(result!.similarity).toBeGreaterThan(0.5)
  })

  it('returns exact match (similarity ~1.0) when an identical vector exists', () => {
    const records = [makeRecord('exact', [1, 2, 3])]
    const result = findBestMatch([1, 2, 3], records, 0.99)
    expect(result).not.toBeNull()
    expect(result!.similarity).toBeCloseTo(1.0)
  })

  it('picks the single record that crosses threshold even when others do not', () => {
    const records = [
      makeRecord('miss', [0, 1, 0]),
      makeRecord('hit', [1, 0, 0]),
    ]
    const result = findBestMatch([1, 0, 0], records, 0.99)
    expect(result?.record.key).toBe('hit')
  })
})
