import { test, expect, type Page } from '@playwright/test'

async function mockLlmConfig(page: Page) {
  await page.route('**/config/llm', async (route) => {
    await route.fulfill({
      json: {
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
      },
    })
  })
}

test('no uploads keeps Mine and Both disabled', async ({ page }) => {
  await mockLlmConfig(page)
  await page.route('**/sessions', async (route) => {
    await route.fulfill({
      json: {
        session_id: 'abc123demo',
        expires_at: Math.floor(Date.now() / 1000) + 1800,
        files: [],
        total_bytes: 0,
        max_session_bytes: 8388608,
        max_files: 3,
      },
    })
  })
  await page.route('**/sessions/abc123demo', async (route) => {
    await route.fulfill({
      json: {
        session_id: 'abc123demo',
        expires_at: Math.floor(Date.now() / 1000) + 1800,
        files: [],
        total_bytes: 0,
        max_session_bytes: 8388608,
        max_files: 3,
      },
    })
  })

  await page.goto('/')
  await page.getByRole('tab', { name: '可信问答' }).click()
  await expect(page.getByRole('radio', { name: /仅我的文档/ })).toBeDisabled()
  await expect(page.getByRole('radio', { name: /合并检索/ })).toBeDisabled()
})

test('query streams an answer', async ({ page }) => {
  await mockLlmConfig(page)
  await page.route('**/sessions', async (route) => {
    await route.fulfill({
      json: {
        session_id: 'abc123demo',
        expires_at: Math.floor(Date.now() / 1000) + 1800,
        files: [],
        total_bytes: 0,
        max_session_bytes: 8388608,
        max_files: 3,
      },
    })
  })
  await page.route('**/query/stream', async (route) => {
    await route.fulfill({
      contentType: 'text/event-stream',
      body: 'data: {"type":"token","text":"Hello from stream"}\n\ndata: {"type":"final","answer":"Hello from stream","status":"answered","refusal_reason":null,"citations":[],"evidence":[],"provider":"deepseek","model":"deepseek-v4-flash"}\n\ndata: [DONE]\n\n',
    })
  })

  await page.goto('/')
  await page.getByRole('tab', { name: '可信问答' }).click()
  await page.getByRole('textbox', { name: /你的问题/ }).fill('什么是 RAG？')
  await page.getByRole('button', { name: '开始可信问答' }).click()
  await expect(page.getByText('Hello from stream')).toBeVisible()
})
