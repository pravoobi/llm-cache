import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { createOpenAIEmbedder } from '../../src/embedders/openai'

describe('createOpenAIEmbedder', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  function mockFetch(embedding: number[]) {
    const mockFetchFn = vi.fn(async (_url: string, _init?: RequestInit) =>
      ({
        ok: true,
        json: async () => ({
          data: [{ embedding }],
        }),
        text: async () => '',
      } as Response)
    )
    vi.stubGlobal('fetch', mockFetchFn)
    return mockFetchFn
  }

  it('calls the correct OpenAI endpoint', async () => {
    const mockFetchFn = mockFetch([0.1, 0.2, 0.3])
    const embedder = createOpenAIEmbedder({ provider: 'openai', apiKey: 'sk-test' })
    await embedder('hello world')

    expect(mockFetchFn).toHaveBeenCalledTimes(1)
    const [url] = mockFetchFn.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('https://api.openai.com/v1/embeddings')
  })

  it('sends the correct request body', async () => {
    const mockFetchFn = mockFetch([0.1, 0.2, 0.3])
    const embedder = createOpenAIEmbedder({
      provider: 'openai',
      apiKey: 'sk-test',
      model: 'text-embedding-3-small',
    })
    await embedder('test input')

    const [, init] = mockFetchFn.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(init.body as string) as { model: string; input: string }
    expect(body.model).toBe('text-embedding-3-small')
    expect(body.input).toBe('test input')
  })

  it('includes the Authorization header with the API key', async () => {
    const mockFetchFn = mockFetch([0.1])
    const embedder = createOpenAIEmbedder({ provider: 'openai', apiKey: 'sk-secret-key' })
    await embedder('test')

    const [, init] = mockFetchFn.mock.calls[0] as [string, RequestInit]
    const headers = init.headers as Record<string, string>
    expect(headers['Authorization']).toBe('Bearer sk-secret-key')
  })

  it('returns the embedding as a number[]', async () => {
    const expectedEmbedding = [0.1, 0.2, 0.3, 0.4]
    mockFetch(expectedEmbedding)
    const embedder = createOpenAIEmbedder({ provider: 'openai', apiKey: 'sk-test' })
    const result = await embedder('hello')

    expect(Array.isArray(result)).toBe(true)
    expect(result).toEqual(expectedEmbedding)
  })

  it('uses text-embedding-3-small as the default model', async () => {
    const mockFetchFn = mockFetch([0.1])
    const embedder = createOpenAIEmbedder({ provider: 'openai', apiKey: 'sk-test' })
    await embedder('test')

    const [, init] = mockFetchFn.mock.calls[0] as [string, RequestInit]
    const body = JSON.parse(init.body as string) as { model: string }
    expect(body.model).toBe('text-embedding-3-small')
  })

  it('throws if the API response is not ok', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({
        ok: false,
        status: 401,
        text: async () => 'Unauthorized',
      }))
    )
    const embedder = createOpenAIEmbedder({ provider: 'openai', apiKey: 'bad-key' })
    await expect(embedder('test')).rejects.toThrow('401')
  })

  it('throws if apiKey is not provided', () => {
    expect(() =>
      createOpenAIEmbedder({ provider: 'openai' })
    ).toThrow('apiKey')
  })
})
