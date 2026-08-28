import { FileSearch, Network, ShieldCheck, Waypoints } from 'lucide-react'

const stages = [
  {
    icon: FileSearch,
    title: '中文文档解析',
    text: '保留文件名、页码、章节和稳定 chunk ID，支持 PDF、DOCX、Markdown、TXT 与 HTML。',
  },
  {
    icon: Network,
    title: '混合检索',
    text: 'Jieba BM25 负责关键词召回，BGE-large-zh 负责语义召回，再通过加权 RRF 合并。',
  },
  {
    icon: Waypoints,
    title: '中文重排',
    text: 'Cross-Encoder 对候选证据重新打分，将最相关的切片送给 DeepSeek。',
  },
  {
    icon: ShieldCheck,
    title: '引用校验与拒答',
    text: '关键事实必须绑定可验证引用；证据不足或引用不可信时，系统直接拒答。',
  },
]

export function OverviewTab() {
  return (
    <div className="space-y-5">
      <section className="app-card overflow-hidden">
        <div className="overview-hero p-6 md:p-8">
          <p className="utility-label text-sky-200">ENTERPRISE KNOWLEDGE / TRACEABLE BY DESIGN</p>
          <h2 className="mt-3 max-w-3xl text-3xl font-semibold tracking-tight text-white md:text-4xl">
            企业知识库可信问答
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-300">
            从中文文档进入索引，到混合检索、重排、生成、引用验证，每一步都保留可检查的证据。系统不知道时，会明确说不知道。
          </p>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        {stages.map((stage, index) => {
          const Icon = stage.icon
          return (
            <article key={stage.title} className="app-card p-5">
              <div className="flex items-start gap-4">
                <span className="stage-icon"><Icon className="h-5 w-5" aria-hidden="true" /></span>
                <div>
                  <p className="font-mono text-[11px] text-sky-700">PIPELINE {String(index + 1).padStart(2, '0')}</p>
                  <h3 className="mt-1 font-semibold text-slate-950">{stage.title}</h3>
                  <p className="mt-2 text-sm leading-relaxed text-slate-600">{stage.text}</p>
                </div>
              </div>
            </article>
          )
        })}
      </section>

      <section className="app-card p-5 md:p-6">
        <div className="grid gap-6 md:grid-cols-[0.8fr_1.2fr]">
          <div>
            <p className="utility-label text-slate-500">HOW TO READ THE RESULT</p>
            <h2 className="mt-2 text-xl font-semibold text-slate-950">如何判断一条回答是否可信</h2>
          </div>
          <dl className="grid gap-4 text-sm sm:grid-cols-3">
            <div className="border-l-2 border-emerald-500 pl-4">
              <dt className="font-semibold text-slate-900">回答状态</dt>
              <dd className="mt-1 leading-relaxed text-slate-600">绿色表示通过依据校验，琥珀色表示拒答及具体原因。</dd>
            </div>
            <div className="border-l-2 border-sky-500 pl-4">
              <dt className="font-semibold text-slate-900">答案依据</dt>
              <dd className="mt-1 leading-relaxed text-slate-600">点击引用查看文件、页码、章节和支持该结论的原文。</dd>
            </div>
            <div className="border-l-2 border-slate-400 pl-4">
              <dt className="font-semibold text-slate-900">检索证据</dt>
              <dd className="mt-1 leading-relaxed text-slate-600">展开技术细节，检查 RRF 分数、重排分数和候选顺序。</dd>
            </div>
          </dl>
        </div>
      </section>
    </div>
  )
}
