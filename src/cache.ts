import type {
  LLMCacheConfig,
  CacheOptions,
  CacheResult,
  StoreAdapter,
  EmbedFn,
  CacheEntry,
  EmbeddingRecord,
} from './types'
import { createEmbedder } from './embedders/index'
import { memoryStore } from './stores/memory'
import { hashPrompt } from './hash'
import { findBestMatch } from './similarity'
import { computeExpiresAt } from './utils/ttl'

interface LifetimeStats {
  hits: number
  misses: number
  similarities: number[]
  seenNamespaces: Set<string>
}

function normalizePrompt(prompt: string): string {
  return prompt.trim().replace(/\s+/g, ' ')
}

function resolveStore(cfg: LLMCacheConfig): StoreAdapter {
  if (!cfg.store || cfg.store === 'memory') return memoryStore()
  return cfg.store
}

function resolveEmbedder(cfg: LLMCacheConfig): EmbedFn {
  if (typeof cfg.embedder === 'function') return cfg.embedder
  return createEmbedder(cfg.embedder)
}

export function createCache(config: LLMCacheConfig) {
  const store = resolveStore(config)
  const embed = resolveEmbedder(config)
  const globalThreshold = config.threshold ?? 0.92
  const globalTtl = config.ttl

  const lifetime: LifetimeStats = {
    hits: 0,
    misses: 0,
    similarities: [],
    seenNamespaces: new Set(),
  }

  async function wrap<T>(
    prompt: string,
    fn: () => Promise<T>,
    options?: CacheOptions
  ): Promise<CacheResult<T>> {
    // bypass short-circuits all cache logic — useful for debugging or force-refresh.
    if (options?.bypass === true) {
      const value = await fn()
      return { value, hit: false, layer: 'miss' }
    }

    const namespace = options?.namespace
    const context = options?.context
    const threshold = options?.threshold ?? globalThreshold
    const ttl = options?.ttl ?? globalTtl

    if (namespace !== undefined) lifetime.seenNamespaces.add(namespace)

    const normalized = normalizePrompt(prompt)
    const key = hashPrompt(namespace, context, normalized)

    // Context is combined with namespace for embedding scope so that two prompts
    // with identical text but different contexts never produce a semantic hit.
    // The combined key is stored in EmbeddingRecord.namespace.
    const embeddingNamespace =
      context !== undefined ? `${namespace ?? ''}__ctx__${context}` : namespace

    // --- Step 1: exact cache lookup ---
    try {
      const cached = await store.get(key)
      if (cached !== null) {
        lifetime.hits++
        const result: CacheResult<T> = {
          value: cached.response as T,
          hit: true,
          layer: 'exact',
          ...(namespace !== undefined ? { namespace } : {}),
        }
        config.onHit?.(result as CacheResult<unknown>)
        return result
      }
    } catch (err) {
      config.onError?.(err instanceof Error ? err : new Error(String(err)))
      lifetime.misses++
      config.onMiss?.(prompt)
      const value = await fn()
      return { value, hit: false, layer: 'miss' }
    }

    // --- Step 2: embed + semantic lookup ---
    let embedding: number[]
    let records: EmbeddingRecord[]

    try {
      const raw = await embed(normalized)
      embedding = Array.from(raw)
      records = await store.listEmbeddings(embeddingNamespace)
    } catch (err) {
      // Cache infrastructure is unavailable; call fn() as a transparent fallback.
      config.onError?.(err instanceof Error ? err : new Error(String(err)))
      lifetime.misses++
      config.onMiss?.(prompt)
      const value = await fn()
      return { value, hit: false, layer: 'miss' }
    }

    try {
      const match = findBestMatch(embedding, records, threshold)
      if (match !== null) {
        const matchedEntry = await store.get(match.record.key)
        if (matchedEntry !== null) {
          lifetime.hits++
          lifetime.similarities.push(match.similarity)
          const result: CacheResult<T> = {
            value: matchedEntry.response as T,
            hit: true,
            layer: 'semantic',
            similarity: match.similarity,
            matchedPrompt: matchedEntry.prompt,
            ...(namespace !== undefined ? { namespace } : {}),
          }
          config.onHit?.(result as CacheResult<unknown>)
          return result
        }
        // The matched embedding's entry was evicted (e.g., expired); fall through.
      }
    } catch (err) {
      // Similarity search failed; don't block the real call.
      config.onError?.(err instanceof Error ? err : new Error(String(err)))
    }

    // --- Step 3: cache miss — call fn() and store result ---
    lifetime.misses++
    config.onMiss?.(prompt)

    // Let fn() errors propagate — we never cache failure responses.
    const value = await fn()

    // Streaming responses cannot be cached (the stream is consumed on first read).
    // Wrap the fn() in a non-streaming adapter before passing it to wrap().
    if (
      value instanceof ReadableStream ||
      (typeof value === 'object' &&
        value !== null &&
        (Symbol.asyncIterator in (value as object) || Symbol.iterator in (value as object)) &&
        typeof (value as { text?: unknown }).text !== 'string')
    ) {
      throw new Error(
        '[llm-cache] Streaming responses cannot be cached. ' +
          'Collect the full response before passing fn() to wrap(), ' +
          'or use bypass: true to skip the cache for streaming calls.'
      )
    }

    const now = Date.now()
    const expiresAt = ttl !== undefined ? computeExpiresAt(ttl) : undefined

    const entry: CacheEntry = {
      prompt: normalized,
      response: value,
      embedding,
      createdAt: now,
      // Store the combined embedding namespace so listEmbeddings can scope results
      // correctly when this entry is later used as a semantic match candidate.
      ...(embeddingNamespace !== undefined ? { namespace: embeddingNamespace } : {}),
      ...(expiresAt !== undefined ? { expiresAt } : {}),
    }

    try {
      await store.set(key, entry, ttl)
    } catch (err) {
      // Store write failure shouldn't break the caller — log and continue.
      config.onError?.(err instanceof Error ? err : new Error(String(err)))
    }

    return {
      value,
      hit: false,
      layer: 'miss',
      ...(namespace !== undefined ? { namespace } : {}),
    }
  }

  async function invalidate(
    prompt: string,
    options?: Pick<CacheOptions, 'namespace' | 'context'>
  ): Promise<void> {
    const normalized = normalizePrompt(prompt)
    const key = hashPrompt(options?.namespace, options?.context, normalized)
    await store.delete(key)
  }

  async function flush(namespace?: string): Promise<void> {
    // If the store exposes a native flush method, prefer it for efficiency.
    const storeWithFlush = store as StoreAdapter & { flush?: (ns?: string) => Promise<void> }
    if (typeof storeWithFlush.flush === 'function') {
      await storeWithFlush.flush(namespace)
      return
    }

    // Generic fallback: list all embedding keys and delete them individually.
    const records = await store.listEmbeddings(namespace)
    await Promise.all(records.map((r) => store.delete(r.key)))
  }

  function getStats(): {
    totalEntries: number
    namespaces: string[]
    hitRate: number
    avgSimilarity: number
  } {
    const total = lifetime.hits + lifetime.misses
    const hitRate = total === 0 ? 0 : lifetime.hits / total
    const avgSimilarity =
      lifetime.similarities.length === 0
        ? 0
        : lifetime.similarities.reduce((a, b) => a + b, 0) / lifetime.similarities.length

    return {
      totalEntries: total,
      namespaces: Array.from(lifetime.seenNamespaces),
      hitRate,
      avgSimilarity,
    }
  }

  return { wrap, invalidate, flush, stats: getStats }
}
