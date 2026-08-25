# -*- coding: utf-8 -*-
"""共享 SQLite 连接工厂与 schema 常量（v9.2 收敛点 CON-8）。

原 session/manager.py 与 guardrails/memory.py 各自维护一份逐字相同的
``_connect_db``（WAL 切换 + 2s 快失败 + busy_timeout=30s），连接口径易漂移；
``memory_store`` 建表 DDL 更是在 manager/memory 两模块共 4 处字面量重复
（改表加列漏一处即行为分裂）。此处收敛为单点：

  - connect_db()         统一连接工厂（行为与 v8.13 SQL-1 逐位一致）
  - MEMORY_STORE_DDL     建表常量（单一真源）
  - ensure_memory_store() 幂等建表助手（4 处调用点统一）
"""
import sqlite3


def connect_db(db_path: str):
    """统一 SQLite 连接工厂——WAL + busy_timeout=30s（v8.13 SQL-1 口径）。

    此前各处裸连接默认 busy_timeout≈5s 且无 WAL：并发写（多请求/多模块
    共用 sessions.db）锁冲突抛 OperationalError，被宽 except 吞掉 →
    save_messages/set_checkpoint 静默失败（消息丢失）。统一口径后由 SQLite
    自身在 30s 窗口内等待锁释放，静默丢失路径关闭。

    journal_mode 切换只在 DELETE→WAL 迁移时需要（持久属性）；库已是 WAL
    时查询即返回、零开销。需切换时以 2s 快失败（其他连接占用时切换会
    busy——保持当前模式继续，busy_timeout 已兜底业务写锁）。
    """
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        cur_mode = (conn.execute("PRAGMA journal_mode").fetchone() or ("",))[0]
        if cur_mode and str(cur_mode).lower() != "wal":
            conn.execute("PRAGMA busy_timeout=2000")
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass
            conn.execute("PRAGMA busy_timeout=30000")
    except Exception:
        pass
    return conn


# memory_store 建表常量（原 manager._purge_synth_history / memory._ensure_ltm_schema /
# _save_store / clear_session 四处字面量重复，v9.2 收敛为单一真源）
MEMORY_STORE_DDL = """CREATE TABLE IF NOT EXISTS memory_store (
        session_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (session_id, key))"""


def ensure_memory_store(conn) -> None:
    """幂等建表 memory_store（4 处调用点统一入口）。"""
    conn.execute(MEMORY_STORE_DDL)