import { Check, CircleCheck, Copy, ShieldAlert } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { QueryResponseModel } from '../api/generated'

const markdownComponents = {
  h1: ({ ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1 className="mb-3 mt-4 text-xl font-bold text-slate-950 first:mt-0" {...props} />
  ),
  h2: ({ ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2 className="mb-2 mt-4 text-lg font-semibold text-slate-950 first:mt-0" {...props} />
  ),
  h3: ({ ...props }: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h3 className="mb-2 mt-3 text-base font-semibold text-slate-900 first:mt-0" {...props} />
  ),
  p: ({ ...props }: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p className="mb-3 leading-relaxed last:mb-0" {...props} />
  ),
  ul: ({ ...props }: React.HTMLAttributes<HTMLUListElement>) => (
    <ul className="mb-3 list-disc space-y-1 pl-5" {...props} />
  ),
  ol: ({ ...props }: React.HTMLAttributes<HTMLOListElement>) => (
    <ol className="mb-3 list-decimal space-y-1 pl-5" {...props} />
  ),
  li: ({ ...props }: React.HTMLAttributes<HTMLLIElement>) => <li className="leading-relaxed" {...props} />,
  strong: ({ ...props }: React.HTMLAttributes<HTMLElement>) => (
    <strong className="font-semibold text-slate-950" {...props} />
  ),
  code: ({ className, children, ...props }: React.HTMLAttributes<HTMLElement>) => {
    const isBlock = typeof className === 'string' && className.includes('language-')
    if (isBlock) {
      return (
        <code className={`block overflow-x-auto rounded-lg bg-slate-900 p-3 text-sm text-slate-100 ${className ?? ''}`} {...props}>
          {children}
        </code>
      )
    }
    return (
      <code className="rounded bg-slate-200 px-1 py-0.5 font-mono text-[0.9em] text-slate-900" {...props}>
        {children}
      </code>
    )
  },
  pre: ({ children, ...props }: React.HTMLAttributes<HTMLPreElement>) => (
    <pre className="mb-3 overflow-x-auto rounded-xl bg-slate-900 p-4 font-mono text-sm leading-relaxed text-slate-100" {...props}>
      {children}
    </pre>
  ),
  a: ({ ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a className="font-medium text-blue-700 underline hover:text-blue-900" {...props} />
  ),
  blockquote: ({ ...props }: React.HTMLAttributes<HTMLQuoteElement>) => (
    <blockquote className="mb-3 border-l-4 border-slate-300 pl-4 text-slate-700 italic" {...props} />
  ),
}

export function AnswerPanel({
  answer,
  response,
  isLoading,
  renderMarkdown,
}: {
  answer: string
  response: QueryResponseModel | null
  isLoading: boolean
  /** When false (streaming tokens), show plain text; when true, render markdown. */
  renderMarkdown: boolean
}) {
  const truthfulness = response?.truthfulness
  const refused = response?.status === 'refused'
  const stateLabel = refused ? '知识库暂无依据' : '已根据知识库回答'
  const refusalReason =
    response?.refusal_reason === 'no_relevant_evidence'
      ? '未检索到达到可信阈值的证据'
      : response?.refusal_reason === 'unverified_citations'
        ? '生成内容的引用未通过校验'
        : response?.refusal_reason
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) {
      return
    }
    const t = window.setTimeout(() => setCopied(false), 2000)
    return () => window.clearTimeout(t)
  }, [copied])

  const handleCopy = useCallback(async () => {
    const text = answer.trim()
    if (!text) {
      return
    }
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
    } catch {
      /* clipboard unavailable */
    }
  }, [answer])

  const emptyPlaceholder = isLoading ? '正在检索证据并生成回答…' : '提出问题后，这里会显示带依据的回答。'
  const showCopy = answer.trim().length > 0

  return (
    <section className={`app-card overflow-hidden border ${refused ? 'border-amber-300' : 'border-emerald-200'}`}>
      {response ? (
        <div className={`flex flex-wrap items-center justify-between gap-3 border-b px-5 py-3 ${refused ? 'border-amber-200 bg-amber-50 text-amber-900' : 'border-emerald-100 bg-emerald-50 text-emerald-900'}`}>
          <span className="inline-flex items-center gap-2 text-sm font-semibold">
            {refused ? <ShieldAlert className="h-4 w-4" aria-hidden="true" /> : <CircleCheck className="h-4 w-4" aria-hidden="true" />}
            {stateLabel}
          </span>
          {refusalReason ? <span className="text-xs">{refusalReason}</span> : null}
        </div>
      ) : null}
      <div className="p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <p className="utility-label text-slate-500">GROUNDED RESPONSE</p>
          <h2 className="mt-1 text-lg font-semibold text-slate-950">回答结果</h2>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            aria-label={copied ? '答案已复制' : '复制答案'}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-800 hover:bg-slate-50 disabled:pointer-events-none disabled:opacity-40"
            disabled={!showCopy}
            onClick={() => void handleCopy()}
          >
            {copied ? (
              <>
                <Check className="h-4 w-4 shrink-0 text-emerald-600" aria-hidden="true" />
                已复制
              </>
            ) : (
              <>
                <Copy className="h-4 w-4 shrink-0" aria-hidden="true" />
                复制答案
              </>
            )}
          </button>
        </div>
      </div>
      <div
        className={`min-h-28 rounded-xl p-4 text-left ${refused ? 'bg-amber-50 text-amber-950' : 'bg-slate-50 text-slate-800'}`}
        aria-live={isLoading ? 'polite' : undefined}
      >
        {!answer && !isLoading ? (
          <p className="text-slate-600">{emptyPlaceholder}</p>
        ) : renderMarkdown && answer ? (
          <div className="answer-markdown">
            <ReactMarkdown components={markdownComponents}>{answer}</ReactMarkdown>
          </div>
        ) : (
          <div className="whitespace-pre-wrap leading-relaxed">{answer || emptyPlaceholder}</div>
        )}
      </div>
      {response ? (
        <div className="mt-3 flex flex-wrap gap-3 text-sm text-slate-600">
          <span>{response.provider} / {response.model}</span>
          <span>{Math.round(response.processing_time_ms)} ms</span>
          {response.cached ? <span>缓存命中</span> : null}
        </div>
      ) : null}
      {truthfulness ? (
        <details className="technical-details mt-4">
          <summary>评测信号</summary>
          <div className="mt-3 grid gap-3 text-sm sm:grid-cols-3">
            <div><span className="block text-xs text-slate-500">综合分</span><strong>{truthfulness.score.toFixed(2)}</strong></div>
            <div><span className="block text-xs text-slate-500">NLI 支持度</span><strong>{truthfulness.nli_faithfulness.toFixed(2)}</strong></div>
            <div><span className="block text-xs text-slate-500">引用依据度</span><strong>{truthfulness.citation_groundedness.toFixed(2)}</strong></div>
          </div>
          <p className="mt-3 text-xs text-slate-500">这些分数是离线/调试信号，不代表事实正确性的保证。</p>
        </details>
      ) : null}
      </div>
    </section>
  )
}
