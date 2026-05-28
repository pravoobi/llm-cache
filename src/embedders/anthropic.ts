import type { EmbedderConfig, EmbedFn } from '../types'

// Anthropic's embedding capability is provided through Voyage AI.
const DEFAULT_MODEL = 'voyage-3'
const VOYAGE_ENDPOINT = 'https://api.voyageai.com/v1/embeddings'

export function createAnthropicEmbedder(config: EmbedderConfig): EmbedFn {
  const model = config.model ?? DEFAULT_MODEL
  const apiKey = config.apiKey

  if (!apiKey) {
    throw new Error('[llm-cache] Anthropic embedder requires config.apiKey (Voyage AI API key)')
  }

  return async (text: string): Promise<number[]> => {
    const response = await fetch(VOYAGE_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model, input: [text] }),
    })

    if (!response.ok) {
      const body = (await response.text()).slice(0, 200)
      throw new Error(
        `[llm-cache] Voyage AI embeddings request failed (${response.status}): ${body}`
      )
    }

    const json = (await response.json()) as {
      data: Array<{ embedding: number[] }>
    }

    const embedding = json.data[0]?.embedding
    if (!embedding) {
      throw new Error('[llm-cache] Voyage AI embeddings response missing data[0].embedding')
    }

    return Array.from(embedding)
  }
}
