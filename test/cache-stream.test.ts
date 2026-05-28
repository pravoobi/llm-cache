import { describe, it, expect, vi, beforeEach } from 'vitest'
import { createCache } from '../src/cache'
import type { LLMCacheConfig } from '../src/types'

async function* toStream<T>(items: T[]): AsyncIterable<T> {
  for (const item of items) yield item
}

async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = []
  for await (const chunk of iter) out.push(chunk)
  return out
}

function makeConfig(embedOverride?: () => number[]): LLMCacheConfig {
  let call = 0
  return {
    embedder: () => Promise.resolve(embedOverride ? embedOverride() : [1, 0, 0]),
    onHit: vi.fn(),
    onMiss: vi.fn(),
    onError: vi.fn(),
  }
}

describe('wrapStream', () => {
  let config: LLMCacheConfig
  let cache: ReturnType<typeof createCache>

  beforeEach(() => {
    config = makeConfig()
    cache = createCache(config)
  })

  it('cache miss — yields all chunks from fn() and stores assembled result', async () => {
    const fn = vi.fn(() => toStream(['Hello', ' ', 'world']))
    const { stream, result } = cache.wrapStream('test prompt', fn)

    const chunks = await collect(stream)
    const r = await result

    expect(chunks).toEqual(['Hello', ' ', 'world'])
    expect(fn).toHaveBeenCalledOnce()
    expect(r.hit).toBe(false)
    expect(r.layer).toBe('miss')
  })

  it('exact cache hit — returns synthetic stream without calling fn()', async () => {
    const fn = vi.fn(() => toStream(['Hello', ' ', 'world']))

    // Prime the cache
    const first = cache.wrapStream('test prompt', fn)
    await collect(first.stream)
    await first.result

    // Second call — should hit
    const fn2 = vi.fn(() => toStream(['should not be called']))
    const { stream, result } = cache.wrapStream('test prompt', fn2)

    const chunks = await collect(stream)
    const r = await result

    expect(fn2).not.toHaveBeenCalled()
    expect(r.hit).toBe(true)
    expect(r.layer).toBe('exact')
    // Default reconstruct yields assembled string as single chunk
    expect(chunks).toEqual(['Hello world'])
  })

  it('semantic cache hit — returns synthetic stream for similar prompt', async () => {
    // Both prompts embed to the same vector so similarity = 1.0
    const fn = vi.fn(() => toStream(['cached response']))
    const first = cache.wrapStream('original prompt', fn)
    await collect(first.stream)
    await first.result

    const fn2 = vi.fn(() => toStream(['should not be called']))
    const { stream, result } = cache.wrapStream('similar prompt', fn2, { threshold: 0.5 })

    const chunks = await collect(stream)
    const r = await result

    expect(fn2).not.toHaveBeenCalled()
    expect(r.hit).toBe(true)
    expect(r.layer).toBe('semantic')
    expect(r.similarity).toBeGreaterThan(0.5)
  })

  it('bypass: true — always calls fn(), never reads or writes cache', async () => {
    // Prime with a non-bypass call
    const first = cache.wrapStream('test prompt', () => toStream(['cached']))
    await collect(first.stream)
    await first.result

    const fn = vi.fn(() => toStream(['live response']))
    const { stream, result } = cache.wrapStream('test prompt', fn, { bypass: true })

    const chunks = await collect(stream)
    const r = await result

    expect(fn).toHaveBeenCalledOnce()
    expect(chunks).toEqual(['live response'])
    expect(r.hit).toBe(false)
  })

  it('different namespaces — no cross-namespace hits', async () => {
    const fn1 = vi.fn(() => toStream(['ns-a response']))
    const first = cache.wrapStream('test prompt', fn1, { namespace: 'ns-a' })
    await collect(first.stream)
    await first.result

    const fn2 = vi.fn(() => toStream(['ns-b response']))
    const { stream, result } = cache.wrapStream('test prompt', fn2, { namespace: 'ns-b' })
    const chunks = await collect(stream)
    const r = await result

    expect(fn2).toHaveBeenCalledOnce()
    expect(chunks).toEqual(['ns-b response'])
    expect(r.hit).toBe(false)
  })

  it('custom assemble — stores the assembled value in cache', async () => {
    type Chunk = { content: string }
    const fn = vi.fn(() => toStream<Chunk>([{ content: 'Hello' }, { content: ' world' }]))
    const assemble = (chunks: Chunk[]) => chunks.map((c) => c.content).join('')

    const first = cache.wrapStream('test', fn, { assemble })
    await collect(first.stream)
    await first.result

    // Hit — reconstruct yields the assembled string as one chunk cast to Chunk
    const fn2 = vi.fn(() => toStream<Chunk>([]))
    const { stream } = cache.wrapStream('test', fn2, { assemble })
    const chunks = await collect(stream)

    expect(fn2).not.toHaveBeenCalled()
    // Default reconstruct yields the cached string ('Hello world') as a single Chunk
    expect(chunks).toEqual(['Hello world'])
  })

  it('custom reconstruct — replays hit as multiple chunks', async () => {
    const fn = vi.fn(() => toStream(['a', 'b', 'c']))
    const assemble = (chunks: string[]) => chunks.join('')
    const reconstruct = async function* (cached: unknown) {
      for (const char of cached as string) yield char
    }

    const first = cache.wrapStream('test', fn, { assemble, reconstruct })
    await collect(first.stream)
    await first.result

    const fn2 = vi.fn(() => toStream([]))
    const { stream } = cache.wrapStream('test', fn2, { assemble, reconstruct })
    const chunks = await collect(stream)

    expect(fn2).not.toHaveBeenCalled()
    expect(chunks).toEqual(['a', 'b', 'c'])
  })

  it('fn() throws — error propagates, nothing cached', async () => {
    const fn = vi.fn(async function* () {
      yield 'partial'
      throw new Error('stream failed')
    })

    const { stream } = cache.wrapStream('test prompt', fn)
    await expect(collect(stream)).rejects.toThrow('stream failed')

    // Cache should be empty — second call should call fn again
    const fn2 = vi.fn(() => toStream(['fresh']))
    const { stream: s2, result } = cache.wrapStream('test prompt', fn2)
    await collect(s2)
    await result

    expect(fn2).toHaveBeenCalledOnce()
  })

  it('embedder throws — falls back to fn() stream non-fatally', async () => {
    const badConfig: LLMCacheConfig = {
      embedder: () => Promise.reject(new Error('embed failed')),
      onError: vi.fn(),
    }
    const badCache = createCache(badConfig)

    const fn = vi.fn(() => toStream(['fallback']))
    const { stream, result } = badCache.wrapStream('test prompt', fn)

    const chunks = await collect(stream)
    const r = await result

    expect(chunks).toEqual(['fallback'])
    expect(r.hit).toBe(false)
    expect(fn).toHaveBeenCalledOnce()
    expect(badConfig.onError).toHaveBeenCalled()
  })

  it('onHit callback is called with hit metadata on cache hit', async () => {
    const first = cache.wrapStream('test', () => toStream(['hi']))
    await collect(first.stream)
    await first.result

    const { stream, result } = cache.wrapStream('test', () => toStream([]))
    await collect(stream)
    await result

    expect(config.onHit).toHaveBeenCalledOnce()
    const hitArg = (config.onHit as ReturnType<typeof vi.fn>).mock.calls[0][0]
    expect(hitArg.hit).toBe(true)
    expect(hitArg.layer).toBe('exact')
  })
})
