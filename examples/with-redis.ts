/**
 * Redis-backed store example.
 * Requires: npm install ioredis
 */
import { createCache, redisStore } from '../src/index'
// @ts-expect-error — ioredis is an optional peer dep
import Redis from 'ioredis'

const client = new Redis({ host: 'localhost', port: 6379 })

const cache = createCache({
  embedder: { provider: 'openai', apiKey: process.env['OPENAI_API_KEY'] },
  store: redisStore(client),
  threshold: 0.92,
  ttl: 3600, // 1 hour
  onError: (err) => console.error('Cache error (non-fatal):', err.message),
})

async function callLLM(prompt: string): Promise<string> {
  // Replace with real OpenAI / Anthropic call
  return `Response to: ${prompt}`
}

async function main() {
  const result = await cache.wrap(
    'Summarize the key differences between REST and GraphQL',
    () => callLLM('Summarize the key differences between REST and GraphQL'),
    { namespace: 'docs-team' }
  )

  console.log('hit:', result.hit, '| layer:', result.layer)
  console.log('value:', result.value)

  await cache.flush('docs-team')
  await client.quit()
}

main().catch(console.error)
