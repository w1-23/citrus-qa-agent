"""
Strategy Cards — 可向量化的执行策略、输出格式、审查规则卡片库
与 SkillTree 共用 fastembed 向量空间，按查询语义自动匹配
"""
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

_CARDS: List[dict] = [
    # ═══════════════ 检索策略 (retrieval) ═══════════════
    {"id": "retrieval_001", "type": "retrieval", "name": "多源并行检索",
     "description": "同时使用RAG+Web+Academic三源检索",
     "prompt": "使用citrus_rag_search、web_search、multi_search同时检索，不串行等待。检索英文关键词。"},
    {"id": "retrieval_002", "type": "retrieval", "name": "高引优先检索",
     "description": "优先检索高引用论文(Nature/Science/Cell/PNAS)",
     "prompt": "检索时优先使用web_search查找高影响因子期刊(Nature/Science/Cell/PNAS/Plant Cell)的最新文献。"},
    {"id": "retrieval_003", "type": "retrieval", "name": "时效优先检索",
     "description": "优先获取近3年文献",
     "prompt": "检索时添加年份限定词(recent/2022-2025)，优先获取近3年最新文献。如无结果再放宽年限。"},
    {"id": "retrieval_004", "type": "retrieval", "name": "英文关键词检索",
     "description": "翻译中文查询→英文关键词→精准检索",
     "prompt": "将查询翻译为英文关键词(3-5个词)，去掉停用词，用空格连接后检索。避免整句中文翻译。"},
    {"id": "retrieval_005", "type": "retrieval", "name": "多角度分解检索",
     "description": "将复杂查询拆为2-3个角度分别检索",
     "prompt": "将复杂查询分解为2-3个子角度(如机制角度+防控角度+基因组角度)，每个角度生成独立英文查询词。"},
    {"id": "retrieval_006", "type": "retrieval", "name": "文献充足则停",
     "description": "检索到足够文献(>=5篇有具体数据)→停止继续检索",
     "prompt": "如果已检索到>=5篇包含具体数值和基因/蛋白实体的文献，不要继续检索，直接返回结果。"},
    {"id": "retrieval_007", "type": "retrieval", "name": "全文优先",
     "description": "优先返回有完整text_preview的文献",
     "prompt": "在多篇文献中，优先选取text_preview字段最长、包含具体方法和数据的文献返回。"},
    {"id": "retrieval_008", "type": "retrieval", "name": "PubMed专业检索",
     "description": "生物医学问题优先走PubMed",
     "prompt": "涉及基因、蛋白、疾病、药物等生物医学问题时，优先使用multi_search的PubMed源。"},
    {"id": "retrieval_009", "type": "retrieval", "name": "跨领域检索",
     "description": "柑橘问题也可查拟南芥/番茄文献作机制参考",
     "prompt": "如果柑橘相关文献不足，可补充检索拟南芥(Arabidopsis)和番茄(tomato)中的同源机制文献作为参考(标记为'跨物种参考')。"},
    {"id": "retrieval_010", "type": "retrieval", "name": "RAG优先降级",
     "description": "RAG无结果→自动降级到web_search",
     "prompt": "先用citrus_rag_search检索本地文献库。如果返回<span3篇或无具体数据，自动降级到web_search和multi_search补充。"},

    # ═══════════════ 规划策略 (planning) ═══════════════
    {"id": "planning_001", "type": "planning", "name": "综述型规划",
     "description": "读文献+读文件→并行检索→写作→校验",
     "prompt": "规划模式: ①检索文献+读取指定文件(并行,depends_on:[]) ②综合写作(depends_on:[①②]) ③校验引用(depends_on:[③])。读写分离,检索并行。"},
    {"id": "planning_002", "type": "planning", "name": "问答型规划",
     "description": "只需1个retrieve任务,不走多Agent",
     "prompt": "规划模式: 仅1个retrieve-agent任务。不解构、不分步、不加write-agent。goal='直接回答用户问题'。depends_on:[]。"},
    {"id": "planning_003", "type": "planning", "name": "数据处理型规划",
     "description": "数据文件→统计→解读→输出",
     "prompt": "规划模式: ①读取数据文件(write-agent+read_local_file) ②统计分析(analyze-agent, depends_on:[①]) ③结果解读+输出(write-agent, depends_on:[②])。"},
    {"id": "planning_004", "type": "planning", "name": "实验设计型规划",
     "description": "检索背景→生成方案→校验",
     "prompt": "规划模式: ①检索文献背景(retrieve-agent) ②生成实验方案(analyze-agent, depends_on:[①]) ③保存(write-agent, depends_on:[②])。"},
    {"id": "planning_005", "type": "planning", "name": "文件处理型规划",
     "description": "读文件→处理内容→保存(不检索文献)",
     "prompt": "规划模式: 1个write-agent完成全部。读文件(用read_local_file/pdf_read)→按指令处理内容→保存(write_local_file)。不创建retrieve-agent。"},
    {"id": "planning_006", "type": "planning", "name": "上下文追问型",
     "description": "基于历史摘要直接回答,不重新检索文献",
     "prompt": "如果有对话历史且当前是追问/确认,不创建retrieve任务。直接从history_summary获取信息回答。depends_on:[]。"},
    {"id": "planning_007", "type": "planning", "name": "多文档对比型",
     "description": "多文件并行读取→交叉对比→写作",
     "prompt": "规划模式: ①并行读取所有文件(write-agent, depends_on:[]) ②检索相关背景(retrieve-agent, depends_on:[]) ③综合比较+写作(write-agent, depends_on:[①,②])。"},

    # ═══════════════ 输出格式 (output) ═══════════════
    {"id": "output_001", "type": "output", "name": "机制解析格式",
     "description": "信号→转录因子→结构基因→代谢物→表型(五段链式)",
     "prompt": "输出格式(机制解析): 按信号感知→信号传导→转录调控→结构基因/酶促级联→代谢流改变→细胞学表型→宏观性状 的链式结构组织。每步标注具体分子和基因名。"},
    {"id": "output_002", "type": "output", "name": "四段式问答格式",
     "description": "核心结论→深度展开→局限与边界→深入探索",
     "prompt": "输出格式(标准问答): 1.核心结论(1-3句) 2.深度展开(≥3维度,含表格) 3.局限与边界(文献确证vs模型推演) 4.深入探索(2-3个追问)。"},
    {"id": "output_003", "type": "output", "name": "数据报告格式",
     "description": "核心数据→区域/时间对比→趋势分析→来源说明",
     "prompt": "输出格式(数据报告): 1.核心数据摘要(3-5条关键数值) 2.对比分析(表格≥3列,含数据来源) 3.趋势解读 4.数据来源与时效性说明。"},
    {"id": "output_004", "type": "output", "name": "对比分析格式",
     "description": "强制表格≥3维对比→逐维展开→总结差异",
     "prompt": "输出格式(对比分析): 先给对比总结(1-2句)→强制Markdown表格(≥3列:对比维度|对象A|对象B|差异解读|文献来源)→逐维度展开→总结核心差异与机制原因。"},
    {"id": "output_005", "type": "output", "name": "综述论文格式",
     "description": "标题→摘要→关键词→引言→章节→结论→局限",
     "prompt": "输出格式(综述): #标题→##摘要(200-300字)→##关键词→##引言→##正文章节(≥4节,每节≥2篇文献引用)→##结论与展望→##局限与边界。优先使用Skill模板的逻辑骨架组织正文。"},
    {"id": "output_006", "type": "output", "name": "实验方案格式",
     "description": "背景(文献依据)→方案(材料/设计/检测)→参数→注意事项",
     "prompt": "输出格式(实验方案): 1.研究背景与假设(文献依据) 2.材料与方法(遗传材料/处理组/对照组/重复数) 3.检测指标与统计方法 4.预期结果 5.注意事项与潜在问题。"},
    {"id": "output_007", "type": "output", "name": "时间线演进格式",
     "description": "早期→近年→最新突破 三段演进",
     "prompt": "输出格式(时间线): 早期(2010-2018): xxx认为... →近年(2019-2023): xxx发现... →最新突破(2024): xxx揭示...。每阶段标注代表文献的期刊名和年份。"},
    {"id": "output_008", "type": "output", "name": "执行摘要格式",
     "description": "2-3句话高度凝练→无需展开",
     "prompt": "输出格式(执行摘要): 仅1-3句话概括核心发现,不展开维度,不列表格。适用于快速查询或确认型回复。标注文献来源。"},
    {"id": "output_009", "type": "output", "name": "术语解释格式",
     "description": "术语→定义→机制简述→引用",
     "prompt": "输出格式(术语解释): 1.术语(中英文) 2.简明定义(1-2句) 3.关键机制简述(2-3句,含具体分子/基因) 4.文献来源。"},
    {"id": "output_010", "type": "output", "name": "FAQ简洁格式",
     "description": "问题→答案(≤50字)→文献来源",
     "prompt": "输出格式(简洁FAQ): 直接回答,不超过100字。不展开机制,不列表格。标注文献来源编号。适用于口语化快速查询。"},

    # ═══════════════ 审查规则 (review) ═══════════════
    {"id": "review_001", "type": "review", "name": "引用完整性检查",
     "description": "检查正文中每个引用编号是否都有对应的参考文献条目",
     "prompt": "审查规则: 逐条检查正文中的[n]引用编号,确认是否在参考文献列表中都有对应的条目。缺失的标记为[missing],错误的标记为[mismatch]。"},
    {"id": "review_002", "type": "review", "name": "数值一致性检查",
     "description": "检查正文中的具体数值是否与检索到的原始数据一致",
     "prompt": "审查规则: 将正文中所有定量数值(百分比、浓度、倍数、p值)与检索结果原文比对。不一致的标记差异幅度,无法验证的标记[数值来源待确认]。"},
    {"id": "review_003", "type": "review", "name": "因果性区分检查",
     "description": "区分正文中的相关性表述 vs 因果性推断,避免过度推断",
     "prompt": "审查规则: 识别正文中所有因果性表述('导致''决定''控制'等),确认其是否有实验证据支撑(CRISPR敲除/过表达验证)。仅有相关性证据的,标记为'关联非因果'。"},
    {"id": "review_004", "type": "review", "name": "领域术语检查",
     "description": "检查柑橘术语使用是否正确,拉丁学名是否规范",
     "prompt": "审查规则: 检查柑橘术语(黄龙病/溃疡病等)是否使用标准中文名,首次出现的物种名是否给出拉丁学名(如甜橙Citrus sinensis),基因/蛋白名格式是否规范。"},
    {"id": "review_005", "type": "review", "name": "数据保真检查",
     "description": "检查是否有'显著''大幅'等模糊词替代了具体数值",
     "prompt": "审查规则: 逐句扫描,标记所有'显著''大幅''明显''众所周知''具有重要意义'等模糊词汇,要求替换为检索结果中的具体数值或删除。"},
    {"id": "review_006", "type": "review", "name": "局限性声明检查",
     "description": "检查局限与边界部分是否充分",
     "prompt": "审查规则: 确认局限与边界部分是否包含: ①文献覆盖面不足的维度 ②仅转录组无蛋白验证的标注 ③仅温室无田间验证的标注 ④仅过表达无敲除验证的标注。"},
    {"id": "review_007", "type": "review", "name": "表格完整性检查",
     "description": "检查对比表格是否>=3列且标注文献来源",
     "prompt": "审查规则: 检查所有Markdown表格是否包含>=3列,每列是否标注文献来源编号。缺失来源的单元格标记[无来源]。"},
    {"id": "review_008", "type": "review", "name": "检索覆盖度检查",
     "description": "检查检索结果是否覆盖了用户问题的所有维度",
     "prompt": "审查规则: 列出用户问题的各个维度,逐一检查检索结果是否覆盖。未覆盖的维度标记[文献缺失:维度名],建议补充检索的英文查询词。"},
]

# ── Card search helpers ──

def get_all_cards() -> List[dict]:
    return _CARDS

def get_cards_by_type(card_type: str) -> List[dict]:
    return [c for c in _CARDS if c["type"] == card_type]

def get_card_texts() -> List[str]:
    """Build embedding texts for all cards."""
    return [
        f"{c['name']} {c['description']} {c['type']} {c['prompt'][:300]}"
        for c in _CARDS
    ]

def build_card_map() -> dict:
    """For fastembed vector indexing."""
    texts = get_card_texts()
    idx_to_id = [c["id"] for c in _CARDS]
    return {"texts": texts, "idx_to_id": idx_to_id}

def load_card_prompt(card_id: str) -> str:
    """Load the full prompt content for a card."""
    for c in _CARDS:
        if c["id"] == card_id:
            return f"## [策略卡片: {c['name']}]\n{c['prompt']}"
    return ""
