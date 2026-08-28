import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ScopeToggle } from './ScopeToggle'

describe('ScopeToggle', () => {
  it('disables Mine and Both until uploads exist', () => {
    render(<ScopeToggle value="global" onChange={vi.fn()} hasUploads={false} />)
    expect(screen.getByRole('radio', { name: /仅我的文档/ })).toBeDisabled()
    expect(screen.getByRole('radio', { name: /合并检索/ })).toBeDisabled()
  })

  it('enables session scopes after upload', async () => {
    const onChange = vi.fn()
    render(<ScopeToggle value="global" onChange={onChange} hasUploads />)
    await userEvent.click(screen.getByRole('radio', { name: /仅我的文档/ }))
    expect(onChange).toHaveBeenCalledWith('session')
  })
})
