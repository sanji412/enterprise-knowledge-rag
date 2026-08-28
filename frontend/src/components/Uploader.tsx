import { useRef, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useQuery } from '@tanstack/react-query'
import { Upload } from 'lucide-react'
import { fetchRuntimeConfig, uploadDocuments, type SessionSummary, type UploadResult } from '../api/client'
import { formatBytes } from '../lib/format'

const ACCEPTED = '.pdf,.docx,.txt,.md,.html'
const MAX_FILE_BYTES = 3 * 1024 * 1024

function resultMessage(result: UploadResult) {
  const messages: Record<string, string> = {
    queued: '上传并建立索引成功。',
    skipped: '检测到重复文件，已跳过。',
    oversize: '文件超过 3 MB 限制。',
    file_count_cap: '已达到当前会话的文件数量上限。',
    session_disk_cap: '已达到当前会话的存储上限。',
    type_mismatch: '文件内容与扩展名不一致。',
  }
  return messages[result.status] ?? messages[result.message] ?? result.message
}

export function Uploader({
  sessionId,
  summary,
  onUploaded,
}: {
  sessionId: string
  summary: SessionSummary | undefined
  onUploaded: () => Promise<unknown>
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [message, setMessage] = useState('')
  const [results, setResults] = useState<UploadResult[]>([])
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  const [chunkStrategyChoice, setChunkStrategyChoice] = useState<string | null>(null)
  const [embeddingProfileChoice, setEmbeddingProfileChoice] = useState<string | null>(null)
  const { data: runtimeConfig } = useQuery({
    queryKey: ['runtime-config'],
    queryFn: fetchRuntimeConfig,
    staleTime: Infinity,
  })
  const chunkStrategy = chunkStrategyChoice
    ?? (runtimeConfig?.chunking_allowed_strategies.includes('zh_structure') ? 'zh_structure' : runtimeConfig?.chunking_default_strategy)
    ?? 'zh_structure'
  const embeddingProfile = embeddingProfileChoice
    ?? (runtimeConfig?.embedding_profiles.st_bge_large_zh ? 'st_bge_large_zh' : runtimeConfig?.embedding_default_profile)
    ?? 'st_bge_large_zh'

  const mutation = useMutation({
    mutationFn: (files: File[]) => uploadDocuments(sessionId, files, { chunkStrategy, embeddingProfile }),
    onSuccess: async (response) => {
      setResults(response.results)
      setMessage('文档上传与索引已完成。')
      await onUploaded()
    },
    onError: (error) => {
      setMessage(error instanceof Error ? error.message : '上传失败，请检查文件或后端服务。')
    },
  })

  const validateFiles = (files: File[]) => {
    if (!summary) {
      return '会话尚未准备完成。'
    }
    const maxFiles = summary?.max_files ?? 3
    const currentFiles = summary?.files.length ?? 0
    if (currentFiles + files.length > maxFiles) {
      return `当前会话还可上传 ${Math.max(0, maxFiles - currentFiles)} 个文件。`
    }
    const oversized = files.find((file) => file.size > MAX_FILE_BYTES)
    if (oversized) {
      return `${oversized.name} 超过 ${formatBytes(MAX_FILE_BYTES)} 限制。`
    }
    return null
  }

  const stageFiles = (fileList: FileList | File[]) => {
    const files = Array.from(fileList)
    const validationError = validateFiles(files)
    if (validationError) {
      setMessage(validationError)
      setPendingFiles([])
      return
    }
    setPendingFiles(files)
    setMessage(`已选择 ${files.length} 个文件，确认后开始索引。`)
  }

  const uploadPendingFiles = () => {
    if (!pendingFiles.length) {
      setMessage('请先选择文件。')
      return
    }
    const validationError = validateFiles(pendingFiles)
    if (validationError) {
      setMessage(validationError)
      return
    }
    mutation.mutate(pendingFiles, {
      onSuccess: async (response) => {
        setResults(response.results)
        setPendingFiles([])
        setMessage('文档上传与索引已完成。')
        await onUploaded()
      },
      onError: (error) => {
        setMessage(error instanceof Error ? error.message : '上传失败，请检查文件或后端服务。')
      },
    })
  }

  const clearSelection = () => {
    setPendingFiles([])
    setMessage('')
    if (inputRef.current) {
      inputRef.current.value = ''
    }
  }

  const canUpload = !!summary && pendingFiles.length > 0 && !mutation.isPending

  return (
    <div>
      {runtimeConfig ? (
        <div className="mb-3 grid gap-3 md:grid-cols-2">
          <label className="block text-sm">
            <span className="mb-1 block font-medium text-slate-700">中文切片策略</span>
            <select
              className="w-full rounded-lg border border-slate-300 bg-white p-2 text-slate-900"
              value={chunkStrategy}
              onChange={(event) => setChunkStrategyChoice(event.target.value)}
            >
              {runtimeConfig.chunking_allowed_strategies.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-sm">
            <span className="mb-1 block font-medium text-slate-700">向量模型</span>
            <select
              className="w-full rounded-lg border border-slate-300 bg-white p-2 text-slate-900"
              value={embeddingProfile}
              onChange={(event) => setEmbeddingProfileChoice(event.target.value)}
            >
              {Object.keys(runtimeConfig.embedding_profiles).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}

      <div
        className="rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 p-8 text-center"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault()
          stageFiles(event.dataTransfer.files)
        }}
      >
        <Upload className="mx-auto mb-3 h-8 w-8 text-blue-600" aria-hidden="true" />
        <p className="font-medium text-slate-900">上传企业文档</p>
        <p className="mt-1 text-sm text-slate-600">支持 PDF、DOCX、TXT、Markdown 和 HTML；确认后开始切片、向量化与索引。</p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          multiple
          className="sr-only"
          onChange={(event) => event.target.files && stageFiles(event.target.files)}
        />
        <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
          <button
            type="button"
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
            disabled={mutation.isPending || !summary}
            onClick={() => inputRef.current?.click()}
          >
            选择文件
          </button>
          <button
            type="button"
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-50"
            disabled={!canUpload}
            onClick={uploadPendingFiles}
          >
            {mutation.isPending ? '正在建立索引…' : '上传并建立索引'}
          </button>
          <button
            type="button"
            className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-100 disabled:opacity-50"
            disabled={mutation.isPending || pendingFiles.length === 0}
            onClick={clearSelection}
          >
            清除选择
          </button>
        </div>
      </div>
      {pendingFiles.length > 0 ? (
        <ul className="mt-3 space-y-1 rounded-lg bg-slate-50 p-3 text-sm text-slate-700">
          {pendingFiles.map((file) => (
            <li key={`${file.name}-${file.size}`}>{file.name}</li>
          ))}
        </ul>
      ) : null}
      {message ? <p className="mt-3 text-sm text-slate-700" aria-live="polite">{message}</p> : null}
      {results.length > 0 ? (
        <ul className="mt-3 space-y-2">
          {results.map((result) => (
            <li key={`${result.filename}-${result.status}`} className="rounded-lg bg-slate-50 p-3 text-sm">
              <span className="font-medium text-slate-900">{result.filename}</span>: {resultMessage(result)}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}
