import { hashPrompt } from '../hash'

export function buildCacheKey(prompt: string, namespace?: string, context?: string): string {
  const normalized = prompt.trim().replace(/\s+/g, ' ')
  return hashPrompt(namespace, context, normalized)
}
