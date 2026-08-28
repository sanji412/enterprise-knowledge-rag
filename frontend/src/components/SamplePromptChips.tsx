const prompts = [
  '员工工作满多久可以享受年假？',
  'ATLAS-X2 出现 E03 错误时应该如何处理？',
  '差旅报销的发票需要在多少天内提交？',
  '极光终端的保修期是多久？',
]

export function SamplePromptChips({ onSelect }: { onSelect: (prompt: string) => void }) {
  return (
    <div>
      <p className="mb-2 text-sm font-medium text-slate-700">试试这些企业问题</p>
      <div className="flex flex-wrap gap-2">
        {prompts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className="rounded-full border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm hover:border-blue-300 hover:text-blue-700"
            onClick={() => onSelect(prompt)}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  )
}
