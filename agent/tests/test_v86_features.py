# -*- coding: utf-8 -*-
"""v8.6 特性测试（对照《深入理解AI Agent》第2/3/4章补充项）:

- BM25 倒排索引 + 持久化缓存（书 §3.2 离线建索引，O8）
- 用户偏好记忆消费点（书 §3.1 偏好追踪，S11）
- 写作技能渐进式披露（书 §2.5/§4.8.2，O6）
- 用户反馈落库（书 §4.6 反馈循环，O7 第一步）
- 提示词版本化快照（书 §6.10.4，O5）
"""
import asyncio
import json
import os
import sys
import uuid

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────
# 1. BM25 倒排索引与持久化（书 §3.2）
# ─────────────────────────────────────────────

def test_bm25_inverted_equivalence():
    print("[BM25] 倒排索引与全量枚举评分逐位等价（O8）")
    from src.retrieval.bm25 import BM25Plus

    corpus = [
        "柑橘黄龙病由 Candidatus Liberibacter asiaticus 引起",
        "柑橘溃疡病 Xanthomonas citri 防治方法研究",
        "黄龙病 CLas 在柑橘木虱体内的传播机制",
        "黄龙病检测 PCR 引物设计 灵敏度 特异性",
        "柑橘种植 水肥管理 产量 品质 影响",
        "黄龙病田间症状 叶片斑驳 果实畸形 防治",
        "溃疡病 铜制剂 抗性品种 田间试验",
        "柑橘采后保鲜 贮藏温度 乙烯 腐烂率",
        "黄龙病防控 综合防治 木虱 砍除病树",
        "柑橘品种 沃柑 砂糖橘 脐橙 栽培技术",
    ]
    bm = BM25Plus()
    bm.fit(corpus)
    check("fit 后倒排索引已构建", bool(bm.inv) and len(bm.inv) > 0)

    queries = ["黄龙病", "柑橘 防治", "PCR 检测 黄龙病", "溃疡病 铜制剂 试验", "不存在的词xyz"]
    for q in queries:
        scan = bm._top_k_scan(_tokenize_for(q), 10)
        inv = bm.top_k(q, k=10)
        check(f"倒排/枚举 命中集合一致: {q[:12]}",
              [i for i, _ in inv] == [i for i, _ in scan],
              f"inv={[i for i, _ in inv]} scan={[i for i, _ in scan]}")
        if inv and scan:
            check(f"倒排/枚举 分数一致: {q[:12]}",
                  all(abs(a - b) < 1e-9 for (_, a), (_, b) in zip(inv, scan)),
                  f"{[round(s, 6) for _, s in inv]} vs {[round(s, 6) for _, s in scan]}")


def _tokenize_for(q):
    from src.retrieval.bm25 import _tokenize
    return _tokenize(q)


def test_bm25_cache_roundtrip():
    print("[BM25] 缓存序列化往返一致 + 指纹敏感（O8）")
    from src.retrieval.bm25 import (
        BM25Plus, bm25_to_cache_dict, bm25_from_cache_dict, BM25_CACHE_FORMAT,
    )

    corpus = ["柑橘黄龙病 CLas 传播机制", "溃疡病防治 铜制剂 品种抗性",
              "采后保鲜 温度 乙烯", "黄龙病 PCR 检测 特异性"] * 5
    bm = BM25Plus()
    bm.fit(corpus)
    d = bm25_to_cache_dict(bm)
    check("缓存 dict 含格式版本", d.get("format") == BM25_CACHE_FORMAT)
    check("缓存不含 tokenized_corpus（省内存）", "tokenized_corpus" not in d)

    bm2 = bm25_from_cache_dict(d)
    for q in ["黄龙病", "PCR", "采后 乙烯"]:
        r1 = bm.top_k(q, k=5)
        r2 = bm2.top_k(q, k=5)
        check(f"往返后结果一致: {q}", [i for i, _ in r1] == [i for i, _ in r2]
              and all(abs(a - b) < 1e-9 for (_, a), (_, b) in zip(r1, r2)))

    from src.retrieval.multi_retriever import _bm25_fingerprint
    fp1 = _bm25_fingerprint(corpus, 1.5, 0.75, 1.0)
    fp2 = _bm25_fingerprint(corpus, 1.5, 0.75, 1.0)
    fp3 = _bm25_fingerprint(corpus + ["新增文献"], 1.5, 0.75, 1.0)
    fp4 = _bm25_fingerprint(corpus, 2.0, 0.75, 1.0)
    check("同语料同参数 → 指纹稳定", fp1 == fp2)
    check("语料变化 → 指纹失效", fp1 != fp3)
    check("参数变化 → 指纹失效", fp1 != fp4)


# ─────────────────────────────────────────────
# 2. 用户偏好记忆（书 §3.1 偏好追踪）
# ─────────────────────────────────────────────

def test_preferences_store():
    print("[Pref] preference_memory 写入/读取/上限（S11）")
    from src.guardrails.memory import MemoryStore
    from _tmpenv import tmp_path

    ms = MemoryStore()
    tmp = tmp_path("db_pref")
    ms.db_path = str(tmp)
    ms.set_preference("s1", "写作语言", "综述一律使用中文")
    ms.set_preference("s1", "结构偏好", "必须包含局限与边界章节")
    ms.set_preference("s2", "另一会话偏好", "偏好不回串")

    text = ms.get_preferences("s1")
    check("读取包含偏好", "写作语言" in text and "综述一律使用中文" in text, text[:80])
    check("读取包含第二条", "结构偏好" in text)
    check("其它会话偏好不混入", "不回串" not in text)
    check("空会话返回空", ms.get_preferences("s9") == "")

    short = ms.get_preferences("s1", max_chars=30)
    check("max_chars 截断生效", len(short) <= 30, f"len={len(short)}")


def test_preferences_in_context():
    print("[Pref] <user_preferences> 注入 build_human_message（S11）")
    from src.core.context_manager import LoadedContext, build_human_message

    ctx = LoadedContext(session_id="s1", mode="expert", query="写一篇综述")
    hm = build_human_message(ctx)
    check("无偏好 → 不注入块", "<user_preferences>" not in (hm.content or ""))

    ctx.user_preferences = "## 用户偏好\n- 写作语言: 中文"
    hm2 = build_human_message(ctx)
    check("有偏好 → 注入块", "<user_preferences>" in hm2.content)
    check("偏好内容保留", "写作语言" in hm2.content)


# ─────────────────────────────────────────────
# 3. 写作技能渐进式披露（书 §2.5/§4.8.2）
# ─────────────────────────────────────────────

_SKILL_MAP = {
    "sci_writing": {"name": "科研写作", "content": "先写结论再给证据，引用编号[n]。"},
    "review_style": {"name": "综述风格", "content": "按主题组织章节，交叉引用文献。"},
    "abstract_style": {"name": "摘要风格", "content": "背景-方法-结果-结论四段式。"},
}


def test_skill_catalog():
    print("[Skill] 渐进式披露目录（第一层，≤800 字符）")
    from src.core.write_pipeline import _build_skill_catalog
    cat = _build_skill_catalog(_SKILL_MAP)
    check("目录含技能 id", "sci_writing" in cat and "review_style" in cat)
    check("目录含用途摘要", "先写结论再给证据" in cat)
    check("目录 ≤800 字符", len(cat) <= 800, f"len={len(cat)}")
    check("目录含按需加载说明", "skills_used" in cat)
    check("空 map → 空串", _build_skill_catalog({}) == "")


def test_skill_resolve():
    print("[Skill] skills_used 解析（JSON 字段 + use_skill 文本标记）")
    from src.core.write_pipeline import _resolve_skills_used
    plan = {"title": "T", "sections": [], "skills_used": ["review_style", "unknown_id"]}
    ids = _resolve_skills_used(plan, "", _SKILL_MAP)
    check("JSON 声明 → 解析且过滤未知 id", ids == ["review_style"], str(ids))

    plan2 = {"title": "T", "sections": []}
    ids2 = _resolve_skills_used(plan2, '使用 use_skill("abstract_style") 技能', _SKILL_MAP)
    check("文本标记 → 解析", ids2 == ["abstract_style"], str(ids2))

    ids3 = _resolve_skills_used(plan2, "无标记", _SKILL_MAP)
    check("无声明 → 空（走兜底）", ids3 == [])
    check("无 skill_map → 空", _resolve_skills_used(plan, "", None) == [])


def test_skill_selected_prompt():
    print("[Skill] 选中技能全文拼装（声明优先；未声明=旧行为全量）")
    from src.core.write_pipeline import _build_selected_skill_prompt
    p1 = _build_selected_skill_prompt(_SKILL_MAP, ["review_style"])
    check("声明技能注入", "综述风格" in p1 and "按主题组织章节" in p1)
    check("未声明技能不注入", "四段式" not in p1 and "先写结论" not in p1, p1[:80])

    p2 = _build_selected_skill_prompt(_SKILL_MAP, [])
    check("未声明 → 全量（旧行为零回归）", "科研写作" in p2 and "综述风格" in p2,
          p2[:60])
    check("空 map → 空串", _build_selected_skill_prompt({}, []) == "")


def test_plan_prompt_catalog():
    print("[Skill] Plan 提示词携带目录（默认不改变旧行为）")
    from src.core.write_pipeline import _build_plan_prompt
    p = _build_plan_prompt([{"doi": "1", "title": "x"}], 0)
    check("无目录参数 → 不含目录", "可用写作技能" not in p)
    p2 = _build_plan_prompt([{"doi": "1", "title": "x"}], 0, skill_catalog="CAT-目录")
    check("带目录 → 追加", "CAT-目录" in p2)


class _CaptureLLM:
    """记录收到的 prompt（messages[1] 为 HumanMessage 内容）。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []

    async def ainvoke(self, messages, **kwargs):
        for m in messages:
            if getattr(m, "type", "") == "human":
                self.prompts.append(getattr(m, "content", ""))
        if self._responses:
            return type("R", (), {"content": self._responses.pop(0)})()
        return type("R", (), {"content": "(no more)"})()


def test_stage2_progressive_injection():
    print("[Skill] Stage2 按声明注入选中技能全文（渐进式披露第二层）")
    import tempfile
    from pathlib import Path
    from src.config import PROJECT_ROOT
    from src.core import write_pipeline as wp
    old_parallel = wp.settings.PIPELINE_PARALLEL_SECTIONS
    wp.settings.PIPELINE_PARALLEL_SECTIONS = 1
    try:
        plan = {"title": "T", "sections": [
            {"heading": "一、背景", "points": ["p1"], "target_chars": 400, "refs": []},
            {"heading": "二、方法", "points": ["p2"], "target_chars": 400, "refs": []},
        ]}
        fname = f"prog_{uuid.uuid4().hex[:6]}.md"
        llm = _CaptureLLM(["## 一、背景\n内容A<summary>s</summary>",
                           "## 二、方法\n内容B<summary>s</summary>"])
        r = run(wp.run_stage2_execute(
            llm, plan, [], fname, skill_prompt="",
            skill_map=_SKILL_MAP, skills_used=["review_style"]))
        check("章节生成完成", r["chapters"] == 2, str(r))
        all_prompts = "\n".join(llm.prompts)
        check("注入声明技能全文", "按主题组织章节" in all_prompts, all_prompts[:200])
        check("未声明技能不注入", "四段式" not in all_prompts and "先写结论" not in all_prompts)
        (wp._WORKSPACE_ROOT / fname).unlink(missing_ok=True)
    finally:
        wp.settings.PIPELINE_PARALLEL_SECTIONS = old_parallel


# ─────────────────────────────────────────────
# 4. 用户反馈落库（书 §4.6 反馈循环）
# ─────────────────────────────────────────────

def test_feedback_store():
    print("[Fb] 反馈落库 + 幂等去重 + 统计 + comment 与 feedback.log 双写")
    import sqlite3
    from pathlib import Path
    from src.session.manager import session_manager
    from _tmpenv import tmp_path, tmp_dir

    tmp = tmp_path("db_fb")
    session_manager.db_path = str(tmp)
    ok1 = session_manager.record_feedback("s1", "m1", 1)
    ok2 = session_manager.record_feedback("s1", "m1", 1)      # 重复 → 幂等
    ok3 = session_manager.record_feedback("s1", "m1", -1, comment="有错误")
    check("写入成功", ok1 and ok2 and ok3)
    stats = session_manager.get_feedback_stats()
    check("正向 1 条", stats["positive"] == 1, str(stats))
    check("负向 1 条", stats["negative"] == 1, str(stats))
    check("重复提交未累加", stats["total"] == 2, str(stats))
    # v8.7: comment 落库
    with sqlite3.connect(tmp) as conn:
        row = conn.execute(
            "SELECT comment FROM feedback WHERE session_id='s1' AND rating=-1"
        ).fetchone()
    check("comment 落库", row and row[0] == "有错误", str(row))
    # v8.7: feedback.log 双写（CITRUS_LOG_DIR 由 conftest 指向 tests/.tmp_runner/logs）
    log_root = os.environ.get("CITRUS_LOG_DIR", "")
    fblog = Path(log_root) / "feedback.log" if log_root else Path("feedback.log")
    check("feedback.log 已写入", fblog.exists()
          and "feedback_recorded" in fblog.read_text(encoding="utf-8", errors="replace"),
          f"exists={fblog.exists()} path={fblog}")
    if fblog.exists():
        fb_txt = fblog.read_text(encoding="utf-8", errors="replace")
        check("feedback.log 含 rating/comment", "rating=1" in fb_txt and "有错误" in fb_txt,
              fb_txt[-200:])


def test_retrieval_log():
    print("[Retr] 统一检索过滤日志（合并后：通过+被过滤明细+阈值+耗时）")
    from pathlib import Path
    from datetime import datetime
    from src.logger import log_retrieval
    from _tmpenv import tmp_dir

    candidates = [
        {"paper_id": "p1", "section_name": "intro", "rerank_score": 0.80,
         "text": "alpha content", "_batch": "b1", "doi": "10.1/x"},
        {"paper_id": "p2", "section_name": "methods", "rerank_score": 0.10,
         "text": "beta content", "_batch": "b2", "doi": "10.1/y"},
    ]
    log_retrieval("测试查询", candidates, 0.25,
                  [candidates[0]], [candidates[1]],
                  elapsed=1.23, extra={"mode": "test"})
    log_root = os.environ.get("CITRUS_LOG_DIR", "")
    p = (Path(log_root) / "retrieval" / f"retrieval_{datetime.now():%Y-%m-%d}.log"
         if log_root else Path("retrieval_test.log"))
    check("检索日志文件存在", p.exists(), str(p))
    if not p.exists():
        return
    content = p.read_text(encoding="utf-8", errors="replace")
    check("记录通过明细", "PASSED" in content and "p1" in content, content[-300:])
    check("记录被过滤明细", "FILTERED" in content and "p2" in content)
    check("记录阈值/耗时/模式", "threshold=0.2500" in content
          and "elapsed=1.23s" in content and "mode=test" in content)


def test_feedback_endpoint():
    print("[Fb] POST /api/v2/feedback 端点（rating 校验 + 落库）")
    from src.api.main import submit_feedback, FeedbackRequest
    from src.session.manager import session_manager
    from _tmpenv import tmp_path
    from fastapi import HTTPException

    tmp = tmp_path("db_fb_api")
    session_manager.db_path = str(tmp)

    resp = run(submit_feedback(FeedbackRequest(
        session_id="s2", message_id="m9", rating=1, comment="")))
    check("端点返回 ok", resp.get("status") == "ok" and resp.get("rating") == 1, str(resp))
    stats = session_manager.get_feedback_stats()
    check("端点数据落库", stats["positive"] == 1, str(stats))

    try:
        run(submit_feedback(FeedbackRequest(session_id="s2", rating=0)))
        check("rating=0 被拒绝", False)
    except HTTPException as e:
        check("rating=0 被拒绝", e.status_code == 400)


# ─────────────────────────────────────────────
# 5. 提示词版本化快照（书 §6.10.4）
# ─────────────────────────────────────────────

def test_prompt_snapshot():
    print("[Snap] 提示词快照渲染确定且完整（O5）")
    from src.prompts.snapshot import render_all
    from src.config import settings
    # 先行的 test_static_prefix 会把全局 CONTEXT_STATIC_PREFIX 置 False 且不还原
    # （其 finally 硬编码 False）——本测试显式设定并还原，避免顺序依赖
    _saved = settings.CONTEXT_STATIC_PREFIX
    settings.CONTEXT_STATIC_PREFIX = True
    try:
        snap1 = render_all(include_strategy_cards=False)
        snap2 = render_all(include_strategy_cards=False)
    finally:
        settings.CONTEXT_STATIC_PREFIX = _saved
    check("渲染非空", len(snap1) >= 8, f"{len(snap1)} files")
    for name, content in snap1.items():
        check(f"快照非空: {name}", len(content) > 5, f"{len(content)} chars")
    check("两次渲染确定性一致", snap1 == snap2)
    check("system_expert 含角色", "你是" in snap1["system_expert.txt"]
          or "You are" in snap1["system_expert.txt"])
    check("含 retrieve-agent", "检索" in snap1["agent_retrieve-agent.txt"])
