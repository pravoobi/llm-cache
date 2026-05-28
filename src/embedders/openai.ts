import type { EmbedderConfig, EmbedFn } from '../types'

const DEFAULT_MODEL = 'text-embedding-3-small'
const OPENAI_ENDPOINT = 'https://api.openai.com/v1/embeddings'

// Uses raw fetch so the openai SDK peer dep is not required at runtime.
// The returned embedding is normalized to a plain number[] for uniform handling.
export function createOpenAIEmbedder(config: EmbedderConfig): EmbedFn {
  const model = config.model ?? DEFAULT_MODEL
  const apiKey = config.apiKey

  if (!apiKey) {
    throw new Error('[llm-cache] OpenAI embedder requires config.apiKey')
  }

  return async (text: string): Promise<number[]> => {
    const response = await fetch(OPENAI_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model, input: text }),
    })

    if (!response.ok) {
      const body = (await response.text()).slice(0, 200)
      throw new Error(`[llm-cache] OpenAI embeddings request failed (${response.status}): ${body}`)
    }

    const json = (await response.json()) as {
      data: Array<{ embedding: number[] }>
    }

    const embedding = json.data[0]?.embedding
    if (!embedding) {
      throw new Error('[llm-cache] OpenAI embeddings response missing data[0].embedding')
    }

    // Normalize Float32Array or plain array to number[] for uniform downstream use.
    return Array.from(embedding)
  }
}
