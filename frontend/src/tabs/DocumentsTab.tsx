import * as Progress from '@radix-ui/react-progress'
import { Fingerprint } from 'lucide-react'
import { Uploader } from '../components/Uploader'
import { formatBytes, formatTtl } from '../lib/format'
import { useSession } from '../session/SessionContext'

export function DocumentsTab({ onOpenUploadFaq }: { onOpenUploadFaq?: () => void }) {
  const { sessionId, summary, expiresAt, refreshSession, clearSession, isMintingSession, bootstrapPaused } = useSession()
  const usedBytes = summary?.total_bytes ?? 0
  const maxBytes = summary?.max_session_bytes ?? 8 * 1024 * 1024
  const files = summary?.files ?? []
  const maxFiles = summary?.max_files ?? 3
  const bytePercent = Math.min(100, (usedBytes / maxBytes) * 100)
  const filePercent = Math.min(100, (files.length / maxFiles) * 100)

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-5">
      <section className="app-card shrink-0 p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-950">上传企业文档</h2>
            <p className="mt-1 text-sm text-slate-600">
              每次会话最多 3 个文件，单个不超过 3 MB，总计不超过 8 MB；闲置 30 分钟后自动过期。
            </p>
          </div>
          <button
            type="button"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium hover:bg-slate-50"
            disabled={!sessionId || isMintingSession || bootstrapPaused}
            title="创建新会话并替换当前会话中的上传文件。"
            onClick={() => void clearSession()}
          >
            <Fingerprint className="h-4 w-4" aria-hidden="true" />
            新建会话
          </button>
        </div>

        <div className="mt-5 grid gap-4 md:grid-cols-2">
          <div>
            <div className="mb-2 flex justify-between text-sm">
              <span>已用空间</span>
              <span>{formatBytes(usedBytes)} / {formatBytes(maxBytes)}</span>
            </div>
            <Progress.Root className="h-2 overflow-hidden rounded-full bg-slate-100" value={bytePercent}>
              <Progress.Indicator className="h-full bg-blue-600" style={{ width: `${bytePercent}%` }} />
            </Progress.Root>
          </div>
          <div>
            <div className="mb-2 flex justify-between text-sm">
              <span>文件数量</span>
              <span>{files.length} / {maxFiles}</span>
            </div>
            <Progress.Root className="h-2 overflow-hidden rounded-full bg-slate-100" value={filePercent}>
              <Progress.Indicator className="h-full bg-emerald-600" style={{ width: `${filePercent}%` }} />
            </Progress.Root>
          </div>
        </div>
        <p className="mt-4 text-sm text-slate-600">当前会话将在 {formatTtl(expiresAt)} 后过期。</p>
      </section>

      {sessionId ? (
        <section className="app-card shrink-0 p-5">
          <div className="mb-3 flex items-center justify-between gap-3 rounded-lg border border-blue-100 bg-blue-50 p-3 text-sm text-blue-900">
            <span>不了解切片策略或向量模型？</span>
            <button type="button" className="font-semibold underline" onClick={() => onOpenUploadFaq?.()}>
              查看上传说明
            </button>
          </div>
          <Uploader sessionId={sessionId} summary={summary} onUploaded={refreshSession} />
        </section>
      ) : null}

      <section className="app-card flex min-h-[42vh] min-h-0 flex-1 flex-col p-5">
        <h2 className="mb-3 shrink-0 text-lg font-semibold text-slate-950">已建立索引的文件</h2>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {files.length === 0 ? (
            <p className="text-sm text-slate-600">还没有上传文档，请先在上方选择文件并建立索引。</p>
          ) : (
            <ul className="space-y-2">
              {files.map((file) => (
                <li key={file.name} className="flex justify-between rounded-lg bg-slate-50 p-3 text-sm">
                  <span className="font-medium text-slate-900">{file.name}</span>
                  <span className="text-slate-600">{formatBytes(file.size_bytes)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  )
}
