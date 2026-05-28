import type { EmbedderConfig, EmbedFn } from '../types'

const DEFAULT_MODEL = 'Xenova/all-MiniLM-L6-v2'

// Pipeline is lazily loaded on first call to avoid the heavy import overhead
// until actually needed. Downloads the model weights on first use (~80MB).
export function createLocalEmbedder(config: EmbedderConfig): EmbedFn {
  const model = config.model ?? DEFAULT_MODEL
  let pipelinePromise: Promise<unknown> | null = null
  let warned = false

  return async (text: string): Promise<number[]> => {
    if (!warned) {
      console.warn(
        `[llm-cache] Local embedder using ${model}. ` +
          'First call will download model weights — this may take a while.'
      )
      warned = true
    }

    if (pipelinePromise === null) {
      // Dynamic import so the peer dep is only resolved at runtime, never at
      // compile time. Using Function constructor bypasses TypeScript's static
      // import analysis for optional peer deps. The module name '@xenova/transformers'
      // is a hardcoded string literal — it is NOT user-controlled and cannot be
      // influenced by callers, so this is not a code injection vector.
      pipelinePromise = (
        new Function('m', 'return import(m)')('@xenova/transformers') as Promise<{
          pipeline: (task: string, model: string) => Promise<unknown>
        }>
      ).then((m) => m.pipeline('feature-extraction', model))
    }

    const pipe = (await pipelinePromise) as (
      text: string,
      options: { pooling: string; normalize: boolean }
    ) => Promise<{ data: Float32Array }>

    const output = await pipe(text, { pooling: 'mean', normalize: true })
    return Array.from(output.data)
  }
}
