import type {
  LLMCacheConfig,
  CacheOptions,
  CacheResult,
  CacheStreamOptions,
  StreamCacheResult,
  StoreAdapter,
  EmbedFn,
  CacheEntry,
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
      context !== undefined ? JSON.stringify([namespace ?? '', context]) : namespace

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

    // --- Step 2: embed ---
    let embedding: number[]

    try {
      const raw = await embed(normalized)
      embedding = Array.from(raw)
    } catch (err) {
      config.onError?.(err instanceof Error ? err : new Error(String(err)))
      lifetime.misses++
      config.onMiss?.(prompt)
      const value = await fn()
      return { value, hit: false, layer: 'miss' }
    }

    // --- Step 3: similarity search (ANN if available, O(n) scan otherwise) ---
    try {
      const match =
        typeof store.searchSimilar === 'function'
          ? await store.searchSimilar(embedding, threshold, embeddingNamespace)
          : findBestMatch(embedding, await store.listEmbeddings(embeddingNamespace), threshold)
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
      (typeof value === 'object' && value !== null && Symbol.asyncIterator in (value as object))
    ) {
      throw new Error(
        '[llm-cache] Streaming responses cannot be cached via wrap(). ' +
          'Use wrapStream() for streaming LLM calls, or collect the full ' +
          'response before passing fn() to wrap().'
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

  function defaultAssemble<T>(chunks: T[]): unknown {
    if (chunks.length > 0 && chunks.every((c) => typeof c === 'string')) {
      return (chunks as string[]).join('')
    }
    return chunks
  }

  async function* defaultReconstruct<T>(cached: unknown): AsyncIterable<T> {
    yield cached as T
  }

  function wrapStream<T>(
    prompt: string,
    fn: () => AsyncIterable<T>,
    options?: CacheStreamOptions<T>
  ): { stream: AsyncIterable<T>; result: Promise<StreamCacheResult> } {
    const assemble = options?.assemble ?? defaultAssemble<T>
    const reconstruct = options?.reconstruct ?? defaultReconstruct<T>

    let resolveResult!: (r: StreamCacheResult) => void
    const result = new Promise<StreamCacheResult>((res) => {
      resolveResult = res
    })

    async function* generate(): AsyncGenerator<T> {
      if (options?.bypass === true) {
        yield* fn()
        resolveResult({ hit: false, layer: 'miss' })
        return
      }

      const namespace = options?.namespace
      const context = options?.context
      const threshold = options?.threshold ?? globalThreshold
      const ttl = options?.ttl ?? globalTtl

      if (namespace !== undefined) lifetime.seenNamespaces.add(namespace)

      const normalized = normalizePrompt(prompt)
      const key = hashPrompt(namespace, context, normalized)
      const embeddingNamespace =
        context !== undefined ? `${namespace ?? ''}__ctx__${context}` : namespace

      // --- Step 1: exact lookup ---
      try {
        const cached = await store.get(key)
        if (cached !== null) {
          lifetime.hits++
          const streamResult: StreamCacheResult = {
            hit: true,
            layer: 'exact',
            ...(namespace !== undefined ? { namespace } : {}),
          }
          config.onHit?.({ ...streamResult, value: cached.response })
          resolveResult(streamResult)
          yield* reconstruct(cached.response)
          return
        }
      } catch (err) {
        config.onError?.(err instanceof Error ? err : new Error(String(err)))
        lifetime.misses++
        config.onMiss?.(prompt)
        yield* fn()
        resolveResult({ hit: false, layer: 'miss' })
        return
      }

      // --- Step 2: embed ---
      let embedding: number[]

      try {
        const raw = await embed(normalized)
        embedding = Array.from(raw)
      } catch (err) {
        config.onError?.(err instanceof Error ? err : new Error(String(err)))
        lifetime.misses++
        config.onMiss?.(prompt)
        yield* fn()
        resolveResult({ hit: false, layer: 'miss' })
        return
      }

      // --- Step 3: similarity search (ANN if available, O(n) scan otherwise) ---
      try {
        const match =
          typeof store.searchSimilar === 'function'
            ? await store.searchSimilar(embedding, threshold, embeddingNamespace)
            : findBestMatch(embedding, await store.listEmbeddings(embeddingNamespace), threshold)
        if (match !== null) {
          const matchedEntry = await store.get(match.record.key)
          if (matchedEntry !== null) {
            lifetime.hits++
            lifetime.similarities.push(match.similarity)
            const streamResult: StreamCacheResult = {
              hit: true,
              layer: 'semantic',
              similarity: match.similarity,
              matchedPrompt: matchedEntry.prompt,
              ...(namespace !== undefined ? { namespace } : {}),
            }
            config.onHit?.({ ...streamResult, value: matchedEntry.response })
            resolveResult(streamResult)
            yield* reconstruct(matchedEntry.response)
            return
          }
        }
      } catch (err) {
        config.onError?.(err instanceof Error ? err : new Error(String(err)))
      }

      // --- Step 3: miss — stream from fn(), collect chunks, store after completion ---
      lifetime.misses++
      config.onMiss?.(prompt)

      const chunks: T[] = []
      // fn() errors propagate — never cache a partial or failed response.
      for await (const chunk of fn()) {
        chunks.push(chunk)
        yield chunk
      }

      const assembled = assemble(chunks)
      const now = Date.now()
      const expiresAt = ttl !== undefined ? computeExpiresAt(ttl) : undefined
      const entry: CacheEntry = {
        prompt: normalized,
        response: assembled,
        embedding,
        createdAt: now,
        ...(embeddingNamespace !== undefined ? { namespace: embeddingNamespace } : {}),
        ...(expiresAt !== undefined ? { expiresAt } : {}),
      }

      try {
        await store.set(key, entry, ttl)
      } catch (err) {
        config.onError?.(err instanceof Error ? err : new Error(String(err)))
      }

      resolveResult({ hit: false, layer: 'miss', ...(namespace !== undefined ? { namespace } : {}) })
    }

    return { stream: generate(), result }
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

  return { wrap, wrapStream, invalidate, flush, stats: getStats }
}
