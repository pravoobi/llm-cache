import type { CacheEntry } from '../types'

export function isExpired(entry: CacheEntry): boolean {
  if (entry.expiresAt === undefined) return false
  return Date.now() > entry.expiresAt
}

// Returns undefined when ttlSeconds is 0, meaning the entry never expires.
export function computeExpiresAt(ttlSeconds: number): number | undefined {
  if (ttlSeconds === 0) return undefined
  return Date.now() + ttlSeconds * 1000
}
