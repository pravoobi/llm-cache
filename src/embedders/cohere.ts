import type { EmbedderConfig, EmbedFn } from '../types'

const DEFAULT_MODEL = 'embed-english-v3.0'
const COHERE_ENDPOINT = 'https://api.cohere.ai/v1/embed'

export function createCohereEmbedder(config: EmbedderConfig): EmbedFn {
  const model = config.model ?? DEFAULT_MODEL
  const apiKey = config.apiKey

  if (!apiKey) {
    throw new Error('[llm-cache] Cohere embedder requires config.apiKey')
  }

  return async (text: string): Promise<number[]> => {
    const response = await fetch(COHERE_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      // input_type "search_document" is for content being stored/indexed;
      // use "search_query" at query time for best Cohere retrieval quality.
      body: JSON.stringify({ model, texts: [text], input_type: 'search_document' }),
    })

    if (!response.ok) {
      const body = (await response.text()).slice(0, 200)
      throw new Error(
        `[llm-cache] Cohere embeddings request failed (${response.status}): ${body}`
      )
    }

    const json = (await response.json()) as {
      embeddings: number[][]
    }

    const embedding = json.embeddings[0]
    if (!embedding) {
      throw new Error('[llm-cache] Cohere embeddings response missing embeddings[0]')
    }

    return Array.from(embedding)
  }
}
