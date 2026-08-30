# 企业知识库 RAG 消融实验

- 生成时间：2026-08-30T07:25:30.770604+00:00
- 原始执行 Commit：`b9c3cb8c19ef0015a8e0c0754beb87a7500832ea`
- 聚合 Commit：`81eca976c4bed241c75cd4770b2fa026b37e1fd2`
- 聚合工作区 Dirty：`true`

## 总体对比

| 实验 | Provider / Model | 检索 | 重排 | Hit@5 | MRR@5 | 事实覆盖 | 引用准确率 | 拒答准确率 | 端到端 P95(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | deepseek / deepseek-v4-flash | bm25 | False | 1.0000 | 1.0000 | 0.8700 | 0.9773 | 0.9000 | 1798.59 |
| vector | deepseek / deepseek-v4-flash | vector | False | 1.0000 | 0.9600 | 0.8900 | 1.0000 | 0.9333 | 2200.50 |
| hybrid | deepseek / deepseek-v4-flash | hybrid | False | 1.0000 | 1.0000 | 0.9200 | 0.9855 | 0.9333 | 2721.05 |
| hybrid_rerank | deepseek / deepseek-v4-flash | hybrid | True | 1.0000 | 1.0000 | 0.8800 | 0.9697 | 0.9000 | 10474.02 |

## Easy 对比

| 实验 | Provider / Model | 检索 | 重排 | Hit@5 | MRR@5 | 事实覆盖 | 引用准确率 | 拒答准确率 | 端到端 P95(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | deepseek / deepseek-v4-flash | bm25 | False | 1.0000 | 1.0000 | 0.9000 | 1.0000 | 0.9000 | 1797.82 |
| vector | deepseek / deepseek-v4-flash | vector | False | 1.0000 | 1.0000 | 0.9000 | 1.0000 | 0.9000 | 4683.05 |
| hybrid | deepseek / deepseek-v4-flash | hybrid | False | 1.0000 | 1.0000 | 0.9000 | 1.0000 | 0.9000 | 1835.06 |
| hybrid_rerank | deepseek / deepseek-v4-flash | hybrid | True | 1.0000 | 1.0000 | 0.9000 | 1.0000 | 0.9000 | 10503.38 |

## Medium 对比

| 实验 | Provider / Model | 检索 | 重排 | Hit@5 | MRR@5 | 事实覆盖 | 引用准确率 | 拒答准确率 | 端到端 P95(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | deepseek / deepseek-v4-flash | bm25 | False | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 0.8000 | 1435.34 |
| vector | deepseek / deepseek-v4-flash | vector | False | 1.0000 | 0.9500 | 0.9500 | 1.0000 | 1.0000 | 2200.50 |
| hybrid | deepseek / deepseek-v4-flash | hybrid | False | 1.0000 | 1.0000 | 0.9000 | 1.0000 | 0.9000 | 2721.05 |
| hybrid_rerank | deepseek / deepseek-v4-flash | hybrid | True | 1.0000 | 1.0000 | 0.9000 | 1.0000 | 0.9000 | 10282.36 |

## Hard 对比

| 实验 | Provider / Model | 检索 | 重排 | Hit@5 | MRR@5 | 事实覆盖 | 引用准确率 | 拒答准确率 | 端到端 P95(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 | deepseek / deepseek-v4-flash | bm25 | False | 1.0000 | 1.0000 | 0.9500 | 0.9000 | 1.0000 | 2540.68 |
| vector | deepseek / deepseek-v4-flash | vector | False | 1.0000 | 0.9000 | 0.7500 | 1.0000 | 0.9000 | 1942.17 |
| hybrid | deepseek / deepseek-v4-flash | hybrid | False | 1.0000 | 1.0000 | 1.0000 | 0.9333 | 1.0000 | 3115.40 |
| hybrid_rerank | deepseek / deepseek-v4-flash | hybrid | True | 1.0000 | 1.0000 | 0.8000 | 0.8333 | 0.9000 | 6552.61 |

## 单条执行失败

- 无
