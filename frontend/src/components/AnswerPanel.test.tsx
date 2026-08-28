import { render, screen } from '@testing-library/react'
import type { QueryResponseModel } from '../api/generated'
import { AnswerPanel } from './AnswerPanel'

describe('AnswerPanel', () => {
  it('renders refusal as a distinct knowledge-base state', () => {
    const response = {
      query: '火星差旅标准是什么',
      provider: 'deepseek',
      model: 'deepseek-v4-flash',
      answer: '当前知识库中没有足够依据回答该问题',
      processing_time_ms: 18,
      cached: false,
      validation_issues: [],
      citations: [],
      retrieved: [],
      truthfulness: null,
      status: 'refused',
      refusal_reason: 'no_relevant_evidence',
      evidence: [],
    } as QueryResponseModel

    render(
      <AnswerPanel
        answer={response.answer}
        response={response}
        isLoading={false}
        renderMarkdown
      />,
    )

    expect(screen.getByText('知识库暂无依据')).toBeInTheDocument()
    expect(screen.getByText('当前知识库中没有足够依据回答该问题')).toBeInTheDocument()
  })
})
