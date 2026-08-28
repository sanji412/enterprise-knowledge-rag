import { testInternals, type StreamEvent } from './streamQuery'

describe('streamQuery parsing', () => {
  it('parses token and final events', () => {
    expect(
      testInternals.parseSseFrame(
        'data: {"type":"token","text":"Hi"}\n\ndata: {"type":"final","citations":[],"retrieved":[],"truthfulness":null,"provider":"ollama","model":"llama3"}',
      ),
    ).toEqual([
      { type: 'token', text: 'Hi' },
      {
        type: 'final',
        citations: [],
        retrieved: [],
        truthfulness: null,
        provider: 'ollama',
        model: 'llama3',
      },
    ])
  })

  it('uses the server refusal answer instead of streamed draft text', () => {
    const final = {
      type: 'final',
      status: 'refused',
      answer: '当前知识库中没有足够依据回答该问题',
      refusal_reason: 'no_relevant_evidence',
      citations: [],
      evidence: [],
      provider: 'deepseek',
      model: 'deepseek-v4-flash',
    } as Extract<StreamEvent, { type: 'final' }>

    expect(testInternals.resolveFinalAnswer('模型正在生成的草稿', final)).toBe(
      '当前知识库中没有足够依据回答该问题',
    )
  })
})
