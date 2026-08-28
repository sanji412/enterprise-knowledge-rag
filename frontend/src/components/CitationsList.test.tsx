import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { CitationModel } from '../api/generated'
import { CitationsList } from './CitationsList'

describe('CitationsList', () => {
  it('opens structured evidence details for a citation', async () => {
    const citation = {
      raw_id: 'annual-leave',
      chunk_id: 'annual-leave',
      resolved: true,
      title: '员工手册',
      filename: '员工手册.pdf',
      page_number: 2,
      section_title: '年假',
      source: '.pdf',
      evidence_anchor: 'handbook.leave.annual',
      claim_text: '员工工作满 12 个月可享 5 天年假',
      text_preview: '工作满 12 个月可享 5 天年假',
      verification_score: 0.92,
      verification: 'supported',
    } as CitationModel

    render(<CitationsList citations={[citation]} />)
    await userEvent.click(screen.getByRole('button', { name: /员工手册\.pdf/ }))

    expect(screen.getByText('员工手册.pdf')).toBeInTheDocument()
    expect(screen.getAllByText(/第 2 页/).length).toBeGreaterThan(0)
    expect(screen.getByText('年假')).toBeInTheDocument()
    expect(screen.getByText('工作满 12 个月可享 5 天年假')).toBeInTheDocument()
  })
})
