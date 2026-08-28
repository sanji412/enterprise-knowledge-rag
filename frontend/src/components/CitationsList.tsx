import { ChevronDown, FileText, ShieldCheck } from 'lucide-react'
import { useState } from 'react'
import type { CitationModel } from '../api/generated'

function verificationLabel(value: string) {
  if (value === 'supported') return '引用已验证'
  if (value === 'weak_support') return '弱支持'
  return '引用未验证'
}

function citationLocation(citation: CitationModel) {
  return [
    citation.filename || citation.title || citation.source || citation.chunk_id,
    citation.page_number ? `第 ${citation.page_number} 页` : null,
    citation.section_title || null,
  ].filter(Boolean).join(' · ')
}

export function CitationsList({ citations }: { citations: CitationModel[] }) {
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null)

  return (
    <section className="app-card p-5">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
        <div>
          <p className="utility-label text-slate-500">CITATION LEDGER</p>
          <h2 className="mt-1 text-lg font-semibold text-slate-950">答案依据</h2>
        </div>
        <span className="font-mono text-xs text-slate-500">{citations.length} VERIFIED LINKS</span>
      </div>
      {citations.length === 0 ? (
        <p className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
          回答完成后，可在这里逐条核对文件、页码、章节与证据原文。
        </p>
      ) : (
        <ul className="space-y-3">
          {citations.map((citation) => {
            const expanded = selectedChunkId === citation.chunk_id
            const sourceName = citation.filename || citation.title || citation.source || citation.chunk_id
            return (
              <li key={citation.chunk_id} className="evidence-ticket">
                <button
                  type="button"
                  className="flex w-full items-center gap-3 p-4 text-left"
                  aria-expanded={expanded}
                  onClick={() => setSelectedChunkId(expanded ? null : citation.chunk_id)}
                >
                  <span className="evidence-ticket-icon"><FileText className="h-4 w-4" aria-hidden="true" /></span>
                  <span className="min-w-0 flex-1">
                    <span className="block font-medium text-slate-950">{citationLocation(citation)}</span>
                    <span className="mt-1 block font-mono text-[11px] text-slate-500">{citation.chunk_id}</span>
                  </span>
                  <span className="hidden items-center gap-1.5 text-xs font-medium text-emerald-700 sm:inline-flex">
                    <ShieldCheck className="h-4 w-4" aria-hidden="true" />
                    {verificationLabel(citation.verification)}
                  </span>
                  <ChevronDown className={`h-4 w-4 text-slate-500 transition ${expanded ? 'rotate-180' : ''}`} aria-hidden="true" />
                </button>
                {expanded ? (
                  <div className="border-t border-slate-200 bg-slate-50 px-4 py-4">
                    <dl className="grid gap-3 text-sm sm:grid-cols-3">
                      <div>
                        <dt className="text-xs text-slate-500">文件</dt>
                        <dd className="mt-1 font-medium text-slate-900">{sourceName}</dd>
                      </div>
                      <div>
                        <dt className="text-xs text-slate-500">页码</dt>
                        <dd className="mt-1 font-medium text-slate-900">{citation.page_number ? `第 ${citation.page_number} 页` : '—'}</dd>
                      </div>
                      <div>
                        <dt className="text-xs text-slate-500">章节</dt>
                        <dd className="mt-1 font-medium text-slate-900">{citation.section_title || '—'}</dd>
                      </div>
                    </dl>
                    <blockquote className="mt-4 border-l-2 border-sky-600 pl-4 text-sm leading-relaxed text-slate-700">
                      {citation.text_preview || '该引用暂未返回证据预览。'}
                    </blockquote>
                    <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-500">
                      <span>{verificationLabel(citation.verification)}</span>
                      <span>verification score {citation.verification_score.toFixed(2)}</span>
                      {citation.evidence_anchor ? <span>anchor {citation.evidence_anchor}</span> : null}
                    </div>
                  </div>
                ) : null}
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
