"""记忆改造 v8.4（ADD-only + 混合信号召回 + 常驻卡片）回归测试."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_ltm_add_only():
    print("[MEM-1] ADD-only 写入（Mem0 v3: 冲突留到检索时解决）")
    from src.guardrails.memory import MemoryStore
    import sqlite3, tempfile, uuid
    from pathlib import Path
    from src.config import PROJECT_ROOT

    ms = MemoryStore()
    tmp = Path(tempfile.mkdtemp()) / "sessions.db"
    # 定向到临时库
    orig = ms.db_path if hasattr(ms, "db_path") else None
    ms.db_path = str(tmp)
    with sqlite3.connect(str(tmp)) as conn:
        ms._ensure_ltm_schema(conn)
        conn.commit()

    ok1 = ms.save_long_term_fact("用户地址", "北京", 0.9, owner_session="s1", source_query="q1")
    ok2 = ms.save_long_term_fact("用户地址", "上海", 0.9, owner_session="s2", source_query="q2")
    check("写入成功", ok1 and ok2)

    with sqlite3.connect(str(tmp)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT fact_key, fact_value FROM ltm_facts WHERE fact_key='用户地址'"
        ).fetchall()
    check("同 key 两版本并存（不覆盖）", len(rows) == 2, f"got {len(rows)}")
    values = {r["fact_value"] for r in rows}
    check("北京/上海都在", values == {"北京", "上海"}, str(values))

    # 低置信度拒绝写入
    ok3 = ms.save_long_term_fact("未核验事实", "xxx", 0.3)
    check("低置信度(<0.5)拒绝写入", not ok3)


def test_ltm_resident_cards():
    print("[MEM-2] 常驻卡片层（高置信度短事实，≤上限自动淘汰）")
    from src.guardrails.memory import MemoryStore
    import sqlite3, tempfile
    from pathlib import Path

    ms = MemoryStore()
    tmp = Path(tempfile.mkdtemp()) / "sessions.db"
    ms.db_path = str(tmp)
    with sqlite3.connect(str(tmp)) as conn:
        ms._ensure_ltm_schema(conn)
        conn.commit()

    for i in range(12):
        ms.save_long_term_fact(f"偏好{i}", f"用户偏好事实{i}", 0.85 + i * 0.01,
                               owner_session="s", source_query="q")
    cards = ms.get_resident_cards()
    check("常驻卡片非空", bool(cards))
    check("常驻卡片 ≤500 字符", len(cards) <= 500, f"len={len(cards)}")
    with sqlite3.connect(str(tmp)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM resident_cards").fetchone()[0]
    check("卡片上限 8 条（自动淘汰）", n <= 8, f"got {n}")
    # 低置信度不进卡片
    ms.save_long_term_fact("低置信事实", "x" * 40, 0.6)
    cards2 = ms.get_resident_cards()
    check("低置信度不进常驻卡片", "x" * 40 not in cards2)


def test_ltm_combined_ranking():
    print("[MEM-3] 混合信号召回（相似度 × 时间衰减置信度，冲突版本都返回）")
    from src.guardrails.memory import MemoryStore
    import sqlite3, tempfile
    from pathlib import Path
    from datetime import datetime, timedelta

    ms = MemoryStore()
    tmp = Path(tempfile.mkdtemp()) / "sessions.db"
    ms.db_path = str(tmp)
    with sqlite3.connect(str(tmp)) as conn:
        ms._ensure_ltm_schema(conn)
        # 旧版本（30 天前）与新版本（现在）并存
        conn.execute(
            "INSERT INTO ltm_facts (fact_key, fact_value, confidence, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("用户所在地", "旧地址A", 0.9,
             (datetime.now() - timedelta(days=30)).isoformat()))
        conn.execute(
            "INSERT INTO ltm_facts (fact_key, fact_value, confidence, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("用户所在地", "新地址B", 0.9, datetime.now().isoformat()))
        conn.commit()

    # 关键词回退路径（不依赖 embedder 加载）
    out = ms._recall_keyword_fallback("用户所在地", top_k=5, owner_session="", max_chars=1500)
    check("关键词回退可召回", bool(out))
    check("新版本排前（时间衰减生效）", out.find("新地址B") < out.find("旧地址A"))


print()
print(f"memory v2 tests: {len(passed)} passed, {len(failed)} failed")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
