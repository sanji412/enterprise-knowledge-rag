import { describe, expect, it } from 'vitest'

import { resolveApiBaseUrlForRuntime } from './client'

describe('resolveApiBaseUrlForRuntime', () => {
  it('uses same-origin for production bundles on any mapped host port', () => {
    expect(resolveApiBaseUrlForRuntime(undefined, true, false)).toBe('')
  })

  it('keeps an explicit API base URL when configured', () => {
    expect(resolveApiBaseUrlForRuntime('https://api.example.com/', true, false)).toBe(
      'https://api.example.com',
    )
  })

  it('uses the absolute MSW endpoint only in tests', () => {
    expect(resolveApiBaseUrlForRuntime(undefined, false, true)).toBe('http://127.0.0.1:8000')
  })
})
