import * as RadioGroup from '@radix-ui/react-radio-group'
import type { QueryRequestModel } from '../api/generated'
import { cn } from '../lib/utils'

type KnowledgeScope = QueryRequestModel['knowledge_scope']

const options: Array<{ value: KnowledgeScope; label: string; helper: string }> = [
  { value: 'global', label: '企业示例库', helper: '检索系统预置的员工制度与产品手册。' },
  { value: 'session', label: '仅我的文档', helper: '只检索当前会话上传并完成索引的文件。' },
  { value: 'both', label: '合并检索', helper: '同时检索企业示例库和当前会话文档。' },
]

export function ScopeToggle({
  value,
  onChange,
  hasUploads,
}: {
  value: KnowledgeScope
  onChange: (value: KnowledgeScope) => void
  hasUploads: boolean
}) {
  return (
    <RadioGroup.Root
      className="grid gap-3 md:grid-cols-3"
      value={value}
      onValueChange={(next) => onChange(next as KnowledgeScope)}
      aria-label="知识库范围"
    >
      {options.map((option) => {
        const disabled = option.value !== 'global' && !hasUploads
        return (
          <RadioGroup.Item
            key={option.value}
            value={option.value}
            disabled={disabled}
            className={cn(
              'rounded-xl border p-4 text-left transition',
              value === option.value ? 'border-blue-500 bg-blue-50' : 'border-slate-200 bg-white',
              disabled && 'cursor-not-allowed opacity-50',
            )}
          >
            <div className="flex items-center gap-3">
              <span
                className={cn(
                  'h-4 w-4 rounded-full border',
                  value === option.value ? 'border-blue-600 bg-blue-600' : 'border-slate-400',
                )}
              />
              <span className="font-medium text-slate-900">{option.label}</span>
            </div>
            <p className="mt-2 text-sm text-slate-600">
              {disabled ? '上传并索引文档后可选。' : option.helper}
            </p>
          </RadioGroup.Item>
        )
      })}
    </RadioGroup.Root>
  )
}
