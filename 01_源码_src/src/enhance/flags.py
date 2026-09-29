# -*- coding: utf-8 -*-
"""增强模块开关（默认全关）。

读取顺序：环境变量 > `agent/enhance.yaml` > 内置默认（关）。
**不读 `config.yaml`、不改 `config.py`** —— 这样实验依赖的配置文件保持逐字节不变。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
AGENT_DIR = HERE.parents[2]          # agent/
YAML_PATH = AGENT_DIR / "enhance.yaml"

_TRUE = {"1", "true", "yes", "on", "y", "t"}

# 内置默认：全部关闭 / 与实验一致
DEFAULTS: dict[str, Any] = {
    "support_check": False,          # 模块1：支持度校验
    "support_threshold": 0.35,       # 交叉编码器相关性阈值（0–1）
    "require_number_match": True,    # 模块1 硬门：论断里的数字必须在所引块出现
    "recheck": False,                # 模块1b：回检闭环
    "recheck_max": 3,                # 回检上限（条/轮）
    "evidence_level": True,          # 模块6：证据等级自动判定（随 support_check 生效）
    "web_fulltext": False,           # 模块2：联网原文块管线
    "web_fulltext_max": 5,           # 最多抓取几条
    "entity_norm": False,            # 模块3：柑橘实体归一
    # 实体归一的**用途开关**：默认只做跨来源对齐 + 冲突检测，**不做查询扩展**。
    # 依据：机制实验 exp3 证明查询扩展会把 gold 挤出（见 FINDINGS.md）。
    "entity_query_expansion": False,
    "entity_alignment": True,        # 跨来源对象对齐（表格行主键，不影响检索）
    "entity_conflict": True,         # 冲突检测（关系比对，不影响检索）
    "dedup_granularity": "doi",      # 模块4：doi（默认，与实验一致）| chunk
}

# 环境变量名 → 内部键
ENV_MAP: dict[str, tuple[str, str]] = {
    "CITRUS_ENABLE_SUPPORT_CHECK": ("support_check", "bool"),
    "CITRUS_SUPPORT_THRESHOLD": ("support_threshold", "float"),
    "CITRUS_SUPPORT_NUMBER_GATE": ("require_number_match", "bool"),
    "CITRUS_ENABLE_RECHECK": ("recheck", "bool"),
    "CITRUS_RECHECK_MAX": ("recheck_max", "int"),
    "CITRUS_ENABLE_EVIDENCE_LEVEL": ("evidence_level", "bool"),
    "CITRUS_ENABLE_WEB_FULLTEXT": ("web_fulltext", "bool"),
    "CITRUS_WEB_FULLTEXT_MAX": ("web_fulltext_max", "int"),
    "CITRUS_ENABLE_ENTITY_NORM": ("entity_norm", "bool"),
    "CITRUS_ENTITY_QUERY_EXPANSION": ("entity_query_expansion", "bool"),
    "CITRUS_ENTITY_ALIGNMENT": ("entity_alignment", "bool"),
    "CITRUS_ENTITY_CONFLICT": ("entity_conflict", "bool"),
    "CITRUS_DEDUP_GRANULARITY": ("dedup_granularity", "str"),
}


def _coerce(kind: str, raw: str):
    if kind == "bool":
        return raw.strip().lower() in _TRUE
    if kind == "int":
        try:
            return int(raw)
        except ValueError:
            return None
    if kind == "float":
        try:
            return float(raw)
        except ValueError:
            return None
    return raw.strip()


def _load_yaml() -> dict:
    if not YAML_PATH.exists():
        return {}
    try:
        import yaml
        d = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
        return d.get("enhance", d) if isinstance(d, dict) else {}
    except Exception:                                          # noqa: BLE001
        return {}


class _Flags:
    """增强开关集合。`FLAGS.support_check` 这种读法在任何时候都安全（默认关）。"""

    def __init__(self) -> None:
        self._d = dict(DEFAULTS)

    def reload(self) -> "_Flags":
        self._d = dict(DEFAULTS)
        y = _load_yaml()
        for k in DEFAULTS:
            if k in y and y[k] is not None:
                self._d[k] = y[k]
        for env, (key, kind) in ENV_MAP.items():
            raw = os.environ.get(env)
            if raw is None or raw == "":
                continue
            v = _coerce(kind, raw)
            if v is not None:
                self._d[key] = v
        # 证据等级判定依附于支持度校验：校验关着就无从判定
        if not self._d["support_check"]:
            self._d["evidence_level"] = False
        if not self._d["support_check"]:
            self._d["recheck"] = False
        return self

    def __getattr__(self, name: str):
        try:
            return self._d[name]
        except KeyError as e:                                  # noqa: BLE001
            raise AttributeError(name) from e

    def as_dict(self) -> dict:
        return dict(self._d)

    def any_on(self) -> bool:
        return bool(self._d["support_check"] or self._d["web_fulltext"]
                    or self._d["entity_norm"]
                    or self._d["dedup_granularity"] != DEFAULTS["dedup_granularity"])


FLAGS = _Flags().reload()


def flags_report() -> dict:
    """给日志/健康检查用：当前开关 + 是否有任何增强被打开。"""
    d = FLAGS.as_dict()
    return {"enhance": d, "any_on": FLAGS.any_on(), "source": "env>enhance.yaml>default"}
