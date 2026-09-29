# -*- coding: utf-8 -*-
"""专利增强包（v10.0）——**默认全部关闭**，关闭时主流程行为与 v9.6.3 逐字节一致。

背景
----
审查员对原申请稿的六条改进建议与四条逻辑矛盾，落成这里的若干模块。
**这些模块不修改任何检索代码**：它们只读证据、只在开关打开时参与生成后的处理，
所以对三个机制实验（exp1/2/3）与系统层对比的结果零影响
（实验只依赖 `src/config.py`、`src/retrieval/*`、`src/engine/{reranker,embedder,hardware}.py`、
`src/tools/search.py`——本包一个都不碰、不改）。

开关
----
一律走**环境变量**（与仓库既有约定 `CITRUS_INTENT_CONFIRM` 一致），
可选被 `agent/enhance.yaml` 覆盖。全部默认关闭：

    CITRUS_ENABLE_SUPPORT_CHECK=1   模块1+6 支持度校验 + 证据等级自动判定
    CITRUS_ENABLE_RECHECK=1         模块1b  回检闭环
    CITRUS_ENABLE_WEB_FULLTEXT=1    模块2   联网原文块管线
    CITRUS_ENABLE_ENTITY_NORM=1     模块3   柑橘实体归一
    CITRUS_DEDUP_GRANULARITY=chunk  模块4   块级去重（默认 doi，与实验一致）

历史记忆证据（[Hn]）**不提供**（用户 2026-09-21 决定）：原稿里 [Hn] 游离于语料指纹
之外的问题，用"不把它写进权利要求"来解决，而不是补一套指纹管理。本包因此没有
history_evidence 模块；既有 [Hn] 代码保持原样不动。
"""
from __future__ import annotations

from .flags import FLAGS, flags_report  # noqa: F401

__all__ = ["FLAGS", "flags_report"]
