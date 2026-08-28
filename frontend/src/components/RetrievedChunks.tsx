import { Binary, ChevronRight, Gauge } from 'lucide-react'
import type { RetrievedChunkModel } from '../api/generated'

function metadataText(metadata: RetrievedChunkModel['metadata'], key: string) {
  const value = metadata?.[key]
  return typeof value === 'string' || typeof value === 'number' ? String(value) : ''
}

export function RetrievedChunks({ chunks }: { chunks: RetrievedChunkModel[] }) {
  return (
    <details className="app-card overflow-hidden">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 p-5 marker:content-none">
        <span>
          <span className="utility-label block text-slate-500">RETRIEVAL TRACE</span>
          <span className="mt-1 block text-lg font-semibold text-slate-950">检索证据</span>
        </span>
        <span className="inline-flex items-center gap-2 text-sm text-slate-600">
          {chunks.length} 个切片
          <ChevronRight className="h-4 w-4" aria-hidden="true" />
        </span>
      </summary>
      <div className="border-t border-slate-200 bg-slate-50 p-5">
        {chunks.length === 0 ? (
          <p className="text-sm text-slate-600">尚未返回检索切片。</p>
        ) : (
          <ol className="space-y-3">
            {chunks.map((chunk, index) => {
              const filename = metadataText(chunk.metadata, 'filename')
              const section = metadataText(chunk.metadata, 'section_title')
              const position = chunk.rerank_position ?? index + 1
              return (
                <li key={chunk.id} className="retrieval-row">
                  <div className="flex min-w-0 flex-1 gap-3">
                    <span className="retrieval-rank">#{position}</span>
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                        <span className="font-medium text-slate-950">{filename || chunk.id}</span>
                        {section ? <span className="text-sm text-slate-500">/ {section}</span> : null}
                      </div>
                      <p className="mt-2 text-sm leading-relaxed text-slate-700">{chunk.preview}</p>
                      <p className="mt-2 break-all font-mono text-[11px] text-slate-400">{chunk.id}</p>
                    </div>
                  </div>
                  <div className="grid shrink-0 grid-cols-2 gap-2 text-xs sm:grid-cols-1">
                    <span className="score-chip"><Binary className="h-3.5 w-3.5" /> RRF {chunk.score.toFixed(3)}</span>
                    <span className="score-chip"><Gauge className="h-3.5 w-3.5" /> Rerank {chunk.cross_encoder_score?.toFixed(3) ?? '—'}</span>
                  </div>
                </li>
              )
            })}
          </ol>
        )}
      </div>
    </details>
  )
}
