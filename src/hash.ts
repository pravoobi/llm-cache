import { createHash } from 'node:crypto'

// SHA-256 of the combined namespace+context+prompt ensures cache key uniqueness
// across all three dimensions without requiring a compound index in stores.
export function hashPrompt(
  namespace: string | undefined,
  context: string | undefined,
  normalizedPrompt: string
): string {
  const input = `${namespace ?? ''}:${context ?? ''}:${normalizedPrompt}`
  return createHash('sha256').update(input, 'utf8').digest('hex')
}
