/**
 * OpenAI embedder + OpenAI chat completion example.
 * Requires: npm install openai
 */
import { createCache, createEmbedder } from '../src/index'
// @ts-expect-error — openai is an optional peer dep
import OpenAI from 'openai'

const apiKey = process.env['OPENAI_API_KEY']
if (!apiKey) throw new Error('OPENAI_API_KEY is required')

const openai = new OpenAI({ apiKey })

const cache = createCache({
  embedder: createEmbedder({ provider: 'openai', model: 'text-embedding-3-small', apiKey }),
  threshold: 0.92,
  ttl: 86400,
  onHit: (r) =>
    console.log(
      `[llm-cache] hit layer=${r.layer}` +
        (r.similarity ? ` similarity=${r.similarity.toFixed(3)}` : '') +
        (r.matchedPrompt ? ` matched="${r.matchedPrompt.slice(0, 50)}"` : '')
    ),
})

async function chat(prompt: string): Promise<string> {
  const resp = await openai.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: prompt }],
  })
  return resp.choices[0]?.message.content ?? ''
}

async function main() {
  // First call: cache miss, real LLM call
  const r1 = await cache.wrap('Explain async/await in JavaScript in one sentence.', () =>
    chat('Explain async/await in JavaScript in one sentence.')
  )
  console.log('r1 hit:', r1.hit, '\n', r1.value)

  // Semantically similar — likely a cache hit depending on embeddings
  const r2 = await cache.wrap(
    'Can you briefly explain how async/await works in JS?',
    () => chat('Can you briefly explain how async/await works in JS?')
  )
  console.log('r2 hit:', r2.hit, 'layer:', r2.layer, '\n', r2.value)

  // Per-user namespace: scopes cache to avoid cross-user hits
  const r3 = await cache.wrap(
    'Explain async/await in JavaScript in one sentence.',
    () => chat('Explain async/await in JavaScript in one sentence.'),
    { namespace: 'user-42' }
  )
  console.log('r3 hit:', r3.hit, '(different namespace, so miss)', '\n', r3.value)

  console.log('\nStats:', cache.stats())
}

main().catch(console.error)
