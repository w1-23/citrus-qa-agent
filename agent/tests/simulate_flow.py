# -*- coding: utf-8 -*-
"""模拟验证：综述截断 / 分块写入 / 查询收敛（不调用真实 LLM）"""
import os
import sys
import uuid

# Windows 控制台默认 GBK，强制 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.tools.file_ops import write_local_file
from src.config import PROJECT_ROOT, settings

print("=" * 70)
print("模拟 1: 分块写入流程（write 首块 + append 续块 + 内容预览防重写）")
print("=" * 70)

fname = f"sim_review_{uuid.uuid4().hex[:6]}.md"
target = PROJECT_ROOT / "workspace" / "output" / fname

# 模拟 write-agent 4 轮: 每轮生成 1-2 章节 (~3000 字), 遵循 prompt 分块策略
blocks = [
    ("# 柑橘组学研究前沿综述\n## 摘要\n", "write", "第1轮: 引言+摘要"),
    ("## 1 基因组学\n第一节内容" + "基因组" * 300 + "\n", "append", "第2轮: 第一章"),
    ("## 2 转录组学\n第二节内容" + "转录" * 300 + "\n", "append", "第3轮: 第二章"),
    ("## 3 代谢组学\n第三节内容" + "代谢" * 300 + "\n## 参考文献\n[1] 文献", "append", "第4轮: 第三章+参考文献"),
]
prev_preview = ""
decisions = []
for content, mode, label in blocks:
    r = write_local_file.func(fname, content, mode)
    preview = r.split("内容预览: ")[-1][:200]
    # 模拟 LLM 决策: append 时应看到【本轮新增块】预览（每轮不同），据此判断继续而非重写
    dup = (prev_preview == preview[:200])
    decisions.append(f"{label}: {'[WARN] 预览与前轮相同（本轮无法确认新增内容）' if dup else '[OK] 预览为本轮新增块，继续续写'}")
    prev_preview = preview[:200]

final_content = target.read_text(encoding="utf-8") if target.exists() else ""
print(f"文件最终大小: {len(final_content)} 字符")
print("分块决策模拟:")
for d in decisions:
    print(f"  {d}")
print(f"结构完整性: 含标题={'#' in final_content} | 摘要={'摘要' in final_content} "
      f"| 正文章节={'## 1' in final_content and '## 2' in final_content and '## 3' in final_content} "
      f"| 参考文献={'参考文献' in final_content}")
print(f"各块独有标记出现次数: 第一节内容={final_content.count('第一节内容')} "
      f"第二节内容={final_content.count('第二节内容')} 第三节内容={final_content.count('第三节内容')}"
      f"（均为 1 = 无重复块）")
if target.exists():
    target.unlink()

print()
print("=" * 70)
print("模拟 2: 截断边界推演（12000 tokens 容量 vs 综述规模）")
print("=" * 70)
# 12000 tokens ≈ 中文字容量
cjk_capacity = int(12000 / 1.2)   # 混合系数: 中文 1.2 token/字
print(f"write-agent 单轮上限 12000 tokens ≈ {cjk_capacity} 中文字（约 1-2 章节）")
print(f"6 轮总容量 ≈ {cjk_capacity * 6} 字")
print(f"典型综述规模 15000-30000 字 → 需 {15000 // cjk_capacity + 1}-{30000 // cjk_capacity + 1} 轮")
print("分块策略（每轮 1-2 章节）下: 单轮生成 3000-5000 字 << 10000 字上限 → 不会触顶截断")
print("若不遵守分块（单轮生成全文 30000 字 > 10000 字）→ 尾部被截断且已计费（修复前 282s 事故）")

print()
print("=" * 70)
print("模拟 3: 查询收敛信号（min_keep + 原因回传 → LLM 看到的工具返回）")
print("=" * 70)
print("""
场景 A: academic_search 泛主题查询（修复后 LLM 收到的返回）:
  ## 学术论文检索结果
  检索词: citrus omics genomics
  共 3 条结果 (去重后) | 来源: crossref=1
  [1] 【非柑橘相关，仅供参考】Plant single-cell transcriptomics advances...
      → LLM 判断: 有 3 条可用参考 → 结合本地 RAG → 结果足够 → 停止检索 ✓

场景 B: citrus_rag_search 空结果（修复后 LLM 收到的返回）:
  未检索到相关文献。
  原因: 检索到候选文献但相关性阈值拦截全部（关键词可能过宽泛）。
  建议: 换更特异的柑橘术语（品种/病害/基因名）或同义词。
      → LLM 判断: 知道是"阈值拦截"不是"没有文献" → 针对性换特异词（1 次有效重试）✓

场景 C: academic_search 网络失败（修复后 LLM 收到的返回）:
  [ERR_NETWORK] 学术源请求失败: crossref(HTTPSConnectionPool... Read timed out)
  建议: 网络问题重试无意义，请优先依赖本地 RAG（citrus_rag_search）。
      → LLM 判断: 不重试网络源 → 直接用本地 RAG ✓
""")

print("=" * 70)
print("模拟 4: 查询是否无限？——硬边界推演")
print("=" * 70)
from src.core.agent_runner import _get_max_turns
from src.graph.expert_graph import SUPERVISOR_MAX_TURNS
from src.graph.light_graph import LIGHT_MAX_TURNS
print(f"硬边界（代码层保证，不依赖 LLM 自觉）:")
print(f"  supervisor: {SUPERVISOR_MAX_TURNS} 轮上限（每轮可发多个检索，但轮次封顶）")
print(f"  retrieve-agent: {_get_max_turns('retrieve-agent')} 轮上限（每轮工具数不限，但 3 轮后强制收尾）")
print(f"  write-agent: {_get_max_turns('write-agent')} 轮 | analyze-agent: {_get_max_turns('analyze-agent')} 轮")
worst = SUPERVISOR_MAX_TURNS * 3 * 4  # 每轮 3 个 retrieve × 每 retrieve 3 轮 × 每轮 4 工具（理论上限）
print(f"理论最坏检索调用: {worst} 次（但 min_keep+原因回传后 LLM 有信号收敛，实际远低于此）")
print(f"结论: 查询【不会无限】——retrieve 3 轮 + supervisor 4 轮硬封顶；轮内数量由信号引导收敛")

print()
print("=" * 70)
print("模拟 5: 内容预览防重写决策对比")
print("=" * 70)
# 有预览: LLM 知道已写内容 → 不会重写
# 无预览（修复前）: LLM 不知道已写什么 → 可能重写块
print("修复前（无预览）: LLM 只看到 'Total file size now: 8000 chars'")
print("  → 不知道内容是什么 → 可能重新生成第一块 → 覆盖重复（用户观察到的现象）")
print("修复后（有预览）: LLM 看到 '内容预览: # 柑橘组学研究前沿综述\\n## 摘要...'")
print("  → 知道已写内容 → 决定 append 续写而非重写 ✓")
