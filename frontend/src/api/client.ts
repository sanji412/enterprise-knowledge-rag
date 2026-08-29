import type { LlmConfigModel, QueryRequestModel, QueryResponseModel, RuntimeConfigModel } from './generated'

export function resolveApiBaseUrlForRuntime(
  raw: string | undefined,
  isProd: boolean,
  isVitest: boolean,
): string {
  if (typeof raw === 'string' && raw.trim() !== '') {
    return raw.trim().replace(/\/$/, '')
  }
  if (!isProd) {
    // Vitest + MSW use absolute handlers on http://127.0.0.1:8000.
    if (isVitest) {
      return 'http://127.0.0.1:8000'
    }
    // npm run dev: empty base → same-origin; vite.config.ts proxies to FastAPI.
    return ''
  }
  // Docker serves the SPA and API from one FastAPI origin. The browser-visible
  // host port may be mapped to any value (8100 by default), so never rewrite it.
  return ''
}

function resolveApiBaseUrl(): string {
  return resolveApiBaseUrlForRuntime(
    import.meta.env.VITE_API_BASE_URL,
    import.meta.env.PROD,
    Boolean(import.meta.env.VITEST),
  )
}

const API_BASE_URL = resolveApiBaseUrl()

export interface SessionFile {
  name: string
  size_bytes: number
}

export interface CreateSessionResponse {
  session_id: string
  expires_at: number
}

export interface DeleteSessionResponse {
  deleted_session_id: string
  session_id: string
}

export interface SessionSummary extends CreateSessionResponse {
  files: SessionFile[]
  total_bytes: number
  max_session_bytes: number
  max_files: number
}

export interface UploadResult {
  filename: string
  status: 'queued' | 'skipped' | 'rejected' | 'failed' | string
  message: string
}

export interface UploadDocumentsResponse extends SessionSummary {
  results: UploadResult[]
}

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown

  constructor(
    message: string,
    status: number,
    detail: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

function apiUrl(path: string) {
  const suffix = path.startsWith('/') ? path : `/${path}`
  return API_BASE_URL ? `${API_BASE_URL}${suffix}` : suffix
}

function readApiKey() {
  return localStorage.getItem('doc-ingestion.api-key') ?? ''
}

async function parseError(response: Response) {
  let detail: unknown
  let rawText = ''
  try {
    rawText = await response.text()
    detail = rawText ? JSON.parse(rawText) : null
  } catch {
    detail = rawText || null
  }
  const message =
    typeof detail === 'object' && detail !== null && 'detail' in detail
      ? String((detail as { detail: unknown }).detail)
      : (typeof detail === 'string' && detail.trim() ? detail : `Request failed with status ${response.status}`)
  return new ApiError(message, response.status, detail)
}

function networkErrorHint(): string {
  const target =
    API_BASE_URL ||
    (typeof window !== 'undefined' ? `${window.location.origin} (vite → API)` : 'the API')
  const connectivity =
    API_BASE_URL === '' && typeof import.meta.env !== 'undefined' && import.meta.env.DEV
      ? 'Start uvicorn on the proxy target (default http://127.0.0.1:8000) while npm run dev is running, '
        + 'set VITE_DEV_API_PROXY_TARGET if the API is elsewhere, '
        + 'or set VITE_API_BASE_URL to bypass the proxy. '
      : 'Start the API (e.g. uvicorn on port 8000), or set VITE_API_BASE_URL at build time. '
  return (
    `Cannot reach ${target}. ${connectivity}` +
    `Session features need DOC_PROFILE=demo and DOC_DEMO_UPLOADS=1 on the server.`
  )
}

/** Thrown when `fetch` fails before a response (offline, wrong host/port, CORS, etc.). */
export function networkFailureError(cause?: unknown): ApiError {
  return new ApiError(networkErrorHint(), 0, cause)
}

async function requestJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const apiKey = readApiKey()
  const headers = new Headers(init.headers)
  if (!(init.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json')
  }
  if (apiKey) {
    headers.set('X-API-Key', apiKey)
  }
  let response: Response
  try {
    response = await fetch(apiUrl(path), { ...init, headers })
  } catch (cause) {
    throw networkFailureError(cause)
  }
  if (!response.ok) {
    const err = await parseError(response)
    if (response.status === 404 && path.startsWith('/sessions')) {
      err.message = `${err.message} If the API is up, enable demo sessions: DOC_PROFILE=demo and DOC_DEMO_UPLOADS=1.`
    }
    throw err
  }
  return response.json() as Promise<T>
}

export function createSession() {
  return requestJson<CreateSessionResponse>('/sessions', { method: 'POST' })
}

export function getSession(sessionId: string) {
  return requestJson<SessionSummary>(`/sessions/${sessionId}`)
}

export function deleteSession(sessionId: string) {
  return requestJson<DeleteSessionResponse>(`/sessions/${sessionId}`, { method: 'DELETE' })
}

export function uploadDocuments(
  sessionId: string,
  files: File[],
  options?: { chunkStrategy?: string; embeddingProfile?: string },
) {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  if (options?.chunkStrategy) {
    formData.append('chunk_strategy', options.chunkStrategy)
  }
  if (options?.embeddingProfile) {
    formData.append('embedding_profile', options.embeddingProfile)
  }
  return requestJson<UploadDocumentsResponse>(`/sessions/${sessionId}/documents`, {
    method: 'POST',
    body: formData,
  })
}

export function queryDocuments(request: QueryRequestModel) {
  return requestJson<QueryResponseModel>('/query', {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

export function fetchLlmConfig() {
  return requestJson<LlmConfigModel>('/config/llm')
}

export function fetchRuntimeConfig() {
  return requestJson<RuntimeConfigModel>('/config/runtime')
}

export { API_BASE_URL }
