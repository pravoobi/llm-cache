import type { EmbedderConfig, EmbedFn } from '../types'
import { createOpenAIEmbedder } from './openai'
import { createAnthropicEmbedder } from './anthropic'
import { createCohereEmbedder } from './cohere'
import { createLocalEmbedder } from './local'

export function createEmbedder(config: EmbedderConfig): EmbedFn {
  switch (config.provider) {
    case 'openai':
      return createOpenAIEmbedder(config)
    case 'anthropic':
      return createAnthropicEmbedder(config)
    case 'cohere':
      return createCohereEmbedder(config)
    case 'local':
      return createLocalEmbedder(config)
    default: {
      // Exhaustiveness check: TypeScript will error if a provider is added to
      // the union but not handled here.
      const _exhaustive: never = config.provider
      throw new Error(`[llm-cache] Unknown embedder provider: ${String(_exhaustive)}`)
    }
  }
}
