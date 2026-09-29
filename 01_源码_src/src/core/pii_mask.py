# -*- coding: utf-8 -*-
"""日志脱敏（v8.4.6 B6）——PII 检测与替换，对照书 §3.1.8 + 实验 3-3。

第一层: 正则快速过滤（邮箱/手机号/身份证/API Key/密钥赋值）。
第二层(可选, 未启用): 本地小模型深度 PII 检测（书实验 3-3 log-sanitization）。

注意: 只脱敏"输出侧"（日志/事件），不修改存储（sessions.db 为用户自有数据，
多用户/外部导出时再按需脱敏）。
"""
import re

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 邮箱
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "<email>"),
    # 中国大陆手机号（11 位，1[3-9] 开头）
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "<phone>"),
    # 身份证号（17 位数字 + 数字/X）
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "<idcard>"),
    # API Key（sk- 前缀等长密钥）
    (re.compile(r"sk-[A-Za-z0-9]{12,}"), "<api_key>"),
    # 密钥/口令赋值（key=value 或 key: value）
    (re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd)\s*[=:]\s*\S+"),
     r"\1=<secret>"),
]


def mask_sensitive(text: str) -> str:
    """对文本做 PII 脱敏（幂等；非字符串原样返回）。"""
    if not isinstance(text, str) or not text:
        return text
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text
