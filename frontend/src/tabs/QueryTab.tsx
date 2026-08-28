import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { ArrowRight, Braces, DatabaseZap, KeyRound, SearchCheck } from 'lucide-react'
import { AnswerPanel } from '../components/AnswerPanel'
import { CitationsList } from '../components/CitationsList'
import { RetrievedChunks } from '../components/RetrievedChunks'
import { SamplePromptChips } from '../components/SamplePromptChips'
import { ScopeToggle } from '../components/ScopeToggle'
import { fetchLlmConfig, queryDocuments } from '../api/client'
import type { QueryRequestModel, QueryResponseModel } from '../api/generated'
import { resolveFinalAnswer, streamQuery } from '../lib/streamQuery'
import { useSession } from '../session/SessionContext'

type KnowledgeScope = QueryRequestModel['knowledge_scope']

const STREAM_PREF_KEY = 'enterprise-rag.query.stream'
const CHINESE_EMBEDDING_PROFILE = 'st_bge_large_zh'

function buildRequest(
  query: string,
  sessionId: string | null,
  scope: KnowledgeScope,
  provider: string,
  model: string,
  providerApiKey: string,
  stream: boolean,
): QueryRequestModel {
  const trimmedKey = providerApiKey.trim()
  return {
    query,
    top_k: 5,
    use_llm: true,
    use_rerank: true,
    stream,
    include_citations: true,
    session_id: sessionId,
    knowledge_scope: scope,
    provider,
    model,
    embedding_profile: CHINESE_EMBEDDING_PROFILE,
    retrieval_mode: 'hybrid',
    ...(trimmedKey ? { provider_api_key: trimmedKey } : {}),
  }
}

function pickDefaultProvider(cfg: {
  default_provider: string
  allowed_models_by_provider: Record<string, string[]>
}) {
  const keys = Object.keys(cfg.allowed_models_by_provider)
  return keys.includes(cfg.default_provider) ? cfg.default_provider : (keys[0] ?? 'deepseek')
}

function pickDefaultModel(
  cfg: {
    default_model_by_provider: Record<string, string>
    allowed_models_by_provider: Record<string, string[]>
  },
  provider: string,
) {
  const models = cfg.allowed_models_by_provider[provider] ?? []
  const configured = cfg.default_model_by_provider[provider]
  return configured && models.includes(configured) ? configured : (models[0] ?? '')
}

export function QueryTab() {
  const { sessionId, hasUploads } = useSession()
  const [queryText, setQueryText] = useState('')
  const [scope, setScope] = useState<KnowledgeScope>('global')
  const [streamingText, setStreamingText] = useState('')
  const [response, setResponse] = useState<QueryResponseModel | null>(null)
  const [message, setMessage] = useState('')
  const answerRef = useRef('')
  const [streamAnswersEnabled, setStreamAnswersEnabled] = useState(() => {
    try {
      return localStorage.getItem(STREAM_PREF_KEY) !== '0'
    } catch {
      return true
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem(STREAM_PREF_KEY, streamAnswersEnabled ? '1' : '0')
    } catch {
      /* storage unavailable */
    }
  }, [streamAnswersEnabled])

  const { data: llmConfig, isLoading: llmConfigLoading, isError: llmConfigError } = useQuery({
    queryKey: ['llm-config'],
    queryFn: fetchLlmConfig,
    staleTime: Infinity,
  })
  const [providerChoice, setProviderChoice] = useState<string | null>(null)
  const [modelChoice, setModelChoice] = useState<string | null>(null)
  const [providerApiKey, setProviderApiKey] = useState('')
  const [useCustomProviderKey, setUseCustomProviderKey] = useState(false)
  const [isRunInProgress, setIsRunInProgress] = useState(false)

  const defaultProvider = llmConfig ? pickDefaultProvider(llmConfig) : ''
  const provider = providerChoice ?? defaultProvider
  const model =
    modelChoice !== null
      ? modelChoice
      : llmConfig && provider
        ? pickDefaultModel(llmConfig, provider)
        : ''
  const providerOptions = llmConfig ? Object.keys(llmConfig.allowed_models_by_provider).sort() : []
  const modelOptions = llmConfig && provider ? llmConfig.allowed_models_by_provider[provider] ?? [] : []
  const serverProviderKeyConfigured = !!llmConfig?.provider_key_configured?.[provider]
  const isDemoMode = !!llmConfig?.demo_mode
  const remoteProvider = provider !== '' && provider !== 'ollama'
  const needsProviderKey = remoteProvider && !serverProviderKeyConfigured && !isDemoMode
  const showModelSelectors = providerOptions.length > 1 || modelOptions.length > 1
  const showKeyInput = remoteProvider && (!isDemoMode || useCustomProviderKey)

  const handleProviderChange = (next: string) => {
    setProviderChoice(next)
    setProviderApiKey('')
    if (llmConfig) {
      setModelChoice(pickDefaultModel(llmConfig, next))
    }
  }

  const fallbackMutation = useMutation({
    mutationFn: (request: QueryRequestModel) => queryDocuments({ ...request, stream: false }),
    onSuccess: (data) => {
      setResponse(data)
      setStreamingText(data.answer)
      answerRef.current = data.answer
      setMessage('流式连接不可用，已自动切换为普通请求。')
    },
    onError: (error) => {
      setMessage(error instanceof Error ? error.message : '查询失败，请检查 API 服务。')
    },
  })

  const submit = async () => {
    const trimmed = queryText.trim()
    if (!trimmed) {
      setMessage('请先输入问题。')
      return
    }
    if (!provider || !model) {
      setMessage('模型配置尚未加载完成。')
      return
    }
    const effectiveProviderKey = showKeyInput ? providerApiKey : ''
    if (needsProviderKey && !effectiveProviderKey.trim()) {
      setMessage('服务器未配置 DeepSeek 密钥，请输入本次会话使用的 API Key。')
      return
    }
    if (scope !== 'global' && !hasUploads) {
      setScope('global')
      setMessage('请先上传并索引文档，当前已切回企业示例库。')
      return
    }

    const request = buildRequest(
      trimmed,
      sessionId,
      scope,
      provider,
      model,
      effectiveProviderKey,
      streamAnswersEnabled,
    )
    answerRef.current = ''
    setStreamingText('')
    setResponse(null)
    setMessage('')
    setIsRunInProgress(true)
    try {
      if (!streamAnswersEnabled) {
        const data = await queryDocuments({ ...request, stream: false })
        setResponse(data)
        setStreamingText(data.answer)
        answerRef.current = data.answer
      } else {
        await streamQuery(request, {
          onToken: (token) => {
            answerRef.current += token
            setStreamingText(answerRef.current)
          },
          onFinal: (final) => {
            const finalAnswer = resolveFinalAnswer(answerRef.current, final)
            answerRef.current = finalAnswer
            setStreamingText(finalAnswer)
            setResponse({
              query: trimmed,
              provider: final.provider,
              model: final.model,
              answer: finalAnswer,
              processing_time_ms: final.processing_time_ms ?? 0,
              cached: final.cached ?? false,
              validation_issues: final.validation_issues ?? [],
              citations: final.citations ?? [],
              retrieved: final.retrieved ?? [],
              truthfulness: final.truthfulness ?? null,
              embedding_profile: final.embedding_profile ?? CHINESE_EMBEDDING_PROFILE,
              status: final.status,
              refusal_reason: final.refusal_reason ?? null,
              evidence: final.evidence ?? [],
            })
          },
        })
      }
    } catch {
      await fallbackMutation.mutateAsync({ ...request, stream: false })
    } finally {
      setIsRunInProgress(false)
    }
  }

  const runDisabled =
    !provider
    || !model
    || llmConfigLoading
    || (needsProviderKey && !providerApiKey.trim())
    || isRunInProgress
    || fallbackMutation.isPending
  const answer = streamingText || response?.answer || ''
  const renderAnswerMarkdown = !isRunInProgress && answer.trim().length > 0

  return (
    <div className="space-y-5">
      <section className="app-card overflow-hidden">
        <div className="query-console-header px-5 py-6 md:px-7">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="utility-label text-sky-200">TRUSTED RAG / HYBRID SEARCH</p>
              <h2 className="mt-2 text-2xl font-semibold text-white">向知识库提问</h2>
              <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-300">
                系统会先检索并重排证据，再由 DeepSeek 生成带引用的中文回答；证据不足时直接拒答。
              </p>
            </div>
            <div className="rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-right text-xs text-slate-300">
              <p className="font-mono text-sky-200">{provider || 'DeepSeek'} / {model || '加载中'}</p>
              <p className="mt-1">BGE-large-zh · Cross-Encoder</p>
            </div>
          </div>
          <div className="mt-5 grid gap-2 sm:grid-cols-4" aria-label="问答处理流程">
            {[
              ['01', '混合召回'],
              ['02', '中文重排'],
              ['03', '依据生成'],
              ['04', '引用校验'],
            ].map(([step, label], index) => (
              <div key={step} className="pipeline-step">
                <span className="font-mono text-[11px] text-sky-300">{step}</span>
                <span className="text-sm font-medium text-white">{label}</span>
                {index < 3 ? <ArrowRight className="ml-auto hidden h-3.5 w-3.5 text-slate-500 sm:block" /> : null}
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-6 p-5 md:p-7">
          <SamplePromptChips
            onSelect={(prompt) => {
              setQueryText(prompt)
              setScope('global')
            }}
          />

          <div>
            <p className="mb-2 text-sm font-medium text-slate-700">知识库范围</p>
            <ScopeToggle value={scope} onChange={setScope} hasUploads={hasUploads} />
          </div>

          {llmConfigError ? (
            <p className="text-sm text-red-700" role="alert">
              无法从 API 读取模型配置，请确认后端服务已启动。
            </p>
          ) : null}

          {llmConfig && showModelSelectors ? (
            <details className="technical-details">
              <summary>模型路由设置</summary>
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <label className="block">
                  <span className="mb-2 block text-sm font-medium text-slate-700">Provider</span>
                  <select
                    className="field-control"
                    value={provider}
                    onChange={(event) => handleProviderChange(event.target.value)}
                  >
                    {providerOptions.map((option) => <option key={option}>{option}</option>)}
                  </select>
                </label>
                <label className="block">
                  <span className="mb-2 block text-sm font-medium text-slate-700">Model</span>
                  <select
                    className="field-control"
                    value={model}
                    onChange={(event) => setModelChoice(event.target.value)}
                  >
                    {modelOptions.map((option) => <option key={option}>{option}</option>)}
                  </select>
                </label>
              </div>
            </details>
          ) : null}

          {remoteProvider && isDemoMode ? (
            <label className="flex cursor-pointer items-center gap-2 text-sm text-slate-700">
              <input
                type="checkbox"
                checked={useCustomProviderKey}
                onChange={(event) => setUseCustomProviderKey(event.target.checked)}
              />
              使用我自己的 DeepSeek API Key
            </label>
          ) : null}

          {showKeyInput ? (
            <label className="block rounded-xl border border-slate-200 bg-slate-50 p-4">
              <span className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-800">
                <KeyRound className="h-4 w-4 text-sky-700" aria-hidden="true" />
                DeepSeek API Key（仅本次会话使用）
              </span>
              <input
                type="password"
                autoComplete="off"
                className="field-control"
                placeholder={serverProviderKeyConfigured ? '可选：覆盖服务器密钥' : '请输入 sk-...'}
                value={providerApiKey}
                onChange={(event) => setProviderApiKey(event.target.value)}
              />
              <span className="mt-2 block text-xs text-slate-500">密钥只保存在当前页面内存中，刷新后清除。</span>
            </label>
          ) : null}

          <label className="block">
            <span className="mb-2 block text-sm font-medium text-slate-700">你的问题</span>
            <textarea
              className="field-control min-h-32 resize-y"
              placeholder="例如：ATLAS-X2 出现 E03 错误时应该如何处理？"
              value={queryText}
              onChange={(event) => setQueryText(event.target.value)}
            />
          </label>

          <div className="flex flex-wrap items-center justify-between gap-4 border-t border-slate-200 pt-5">
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-600">
              <span className="pipeline-chip"><DatabaseZap className="h-3.5 w-3.5" /> BM25 + Vector</span>
              <span className="pipeline-chip"><SearchCheck className="h-3.5 w-3.5" /> Rerank enabled</span>
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={streamAnswersEnabled}
                  onChange={(event) => setStreamAnswersEnabled(event.target.checked)}
                />
                流式输出
              </label>
            </div>
            <button
              type="button"
              className="primary-action"
              disabled={runDisabled}
              onClick={() => void submit()}
            >
              <Braces className="h-4 w-4" aria-hidden="true" />
              {isRunInProgress || fallbackMutation.isPending ? '正在检索证据…' : '开始可信问答'}
            </button>
          </div>
          {message ? <p className="text-sm text-slate-700" aria-live="polite">{message}</p> : null}
        </div>
      </section>

      <AnswerPanel
        answer={answer}
        response={response}
        isLoading={isRunInProgress || fallbackMutation.isPending}
        renderMarkdown={renderAnswerMarkdown}
      />
      <CitationsList citations={response?.citations ?? []} />
      <RetrievedChunks chunks={response?.retrieved ?? []} />
    </div>
  )
}
