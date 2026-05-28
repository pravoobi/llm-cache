/**
 * Minimal example: in-memory store, custom embedder stub.
 * Replace the embedder function with a real embedding API for production use.
 */
import { createCache } from '../src/index'

const cache = createCache({
  // Swap this function for createEmbedder({ provider: 'openai', apiKey: '...' }) in production
  embedder: async (_text: string) => new Array(4).fill(0).map(() => Math.random()) as number[],
  threshold: 0.92,
  ttl: 86400,
  onHit: (r) => console.log(`Cache hit [${r.layer}]${r.similarity ? ` similarity=${r.similarity.toFixed(3)}` : ''}`),
  onMiss: (p) => console.log(`Cache miss: "${p.slice(0, 60)}"`),
})

async function callLLM(prompt: string): Promise<string> {
  // Simulated LLM call
  return `Response to: ${prompt}`
}

async function main() {
  const result1 = await cache.wrap('What is the capital of France?', () =>
    callLLM('What is the capital of France?')
  )
  console.log(result1.value, '| hit:', result1.hit)

  // Exact same prompt — hits the exact layer
  const result2 = await cache.wrap('What is the capital of France?', () =>
    callLLM('What is the capital of France?')
  )
  console.log(result2.value, '| hit:', result2.hit, '| layer:', result2.layer)

  console.log('Stats:', cache.stats())
}

main().catch(console.error)
