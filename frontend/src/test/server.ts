import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'

export const mockSession = {
  session_id: 'abc123demo',
  expires_at: Math.floor(Date.now() / 1000) + 1800,
  files: [],
  total_bytes: 0,
  max_session_bytes: 8 * 1024 * 1024,
  max_files: 3,
}

const mockLlmConfig = {
  default_provider: 'deepseek',
  default_model_by_provider: {
    deepseek: 'deepseek-v4-flash',
  },
  allowed_models_by_provider: {
    deepseek: ['deepseek-v4-flash'],
  },
  provider_key_configured: {
    deepseek: true,
  },
  demo_mode: true,
}
const mockRuntimeConfig = {
  chunking_default_strategy: 'zh_structure',
  chunking_allowed_strategies: ['zh_structure', 'tiktoken'],
  embedding_default_profile: 'st_bge_large_zh',
  embedding_profiles: {
    st_bge_large_zh: { provider: 'sentence_transformers', framework: 'sentence_transformers', model: 'BAAI/bge-large-zh-v1.5', dimension: 1024, options: {} },
  },
}

export const handlers = [
  http.get('http://127.0.0.1:8000/config/llm', () => HttpResponse.json(mockLlmConfig)),
  http.get('http://127.0.0.1:8000/config/runtime', () => HttpResponse.json(mockRuntimeConfig)),
  http.post('http://127.0.0.1:8000/sessions', () => HttpResponse.json(mockSession)),
  http.get('http://127.0.0.1:8000/sessions/:sid', () => HttpResponse.json(mockSession)),
  http.delete('http://127.0.0.1:8000/sessions/:sid', () =>
    HttpResponse.json({ deleted_session_id: 'abc123demo', session_id: 'new123demo' }),
  ),
  http.post('http://127.0.0.1:8000/sessions/:sid/documents', () =>
    HttpResponse.json({
      ...mockSession,
      files: [{ name: 'uploaded-doc.md', size_bytes: 128 }],
      total_bytes: 128,
      results: [{ filename: 'uploaded-doc.md', status: 'queued', message: 'indexed' }],
    }),
  ),
  http.post('http://127.0.0.1:8000/query', () =>
    HttpResponse.json({
      query: 'What is in my file?',
      provider: 'deepseek',
      model: 'deepseek-v4-flash',
      answer: 'The uploaded file says hello. [1]',
      processing_time_ms: 12,
      cached: false,
      validation_issues: [],
      citations: [
        {
          raw_id: '1',
          chunk_id: 'chunk-1',
          resolved: true,
          title: 'uploaded-doc.md',
          source: 'uploaded-doc.md',
          verification_score: 0.95,
          verification: 'supported',
        },
      ],
      retrieved: [],
      status: 'answered',
      refusal_reason: null,
      evidence: [],
      truthfulness: { nli_faithfulness: 1, citation_groundedness: 0.95, uncited_claims: 0, score: 0.97 },
    }),
  ),
  http.post('http://127.0.0.1:8000/query/stream', () => {
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"type":"token","text":"Hello"}\n\n'))
        controller.enqueue(encoder.encode('data: {"type":"token","text":" world"}\n\n'))
        controller.enqueue(
          encoder.encode(
            'data: {"type":"final","answer":"Hello world","status":"answered","refusal_reason":null,"citations":[],"evidence":[],"retrieved":[],"truthfulness":null,"provider":"deepseek","model":"deepseek-v4-flash"}\n\ndata: [DONE]\n\n',
          ),
        )
        controller.close()
      },
    })
    return new HttpResponse(stream, { headers: { 'Content-Type': 'text/event-stream' } })
  }),
]

export const server = setupServer(...handlers)
