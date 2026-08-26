# -*- coding: utf-8 -*-
"""一次性脚本：把 blog/diag 调用点 try/import/except-pass 样板收敛为
safe_blog/safe_diag 直调（仅处理 body 全为 import+blog/diag 调用的 try 块）。"""
import re
import sys

FILES = [
    r"E:\codex_WORKSPACES\Citrus_QA_Agent\agent\src\api\main.py",
    r"E:\codex_WORKSPACES\Citrus_QA_Agent\agent\src\core\write_pipeline.py",
    r"E:\codex_WORKSPACES\Citrus_QA_Agent\agent\src\core\agent_runner.py",
]

IMPORT_RE = re.compile(r"from src\.core\.(business_logger import blog|diag import diag)")
CALL_IMPORT_MAP = {
    "from src.core.business_logger import blog": "from src.core.business_logger import safe_blog",
    "from src.core.diag import diag": "from src.core.diag import safe_diag",
}


def transform(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    out = []
    i = 0
    converted = 0
    skipped = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)try:\s*$", line)
        if not m:
            out.append(line)
            i += 1
            continue
        indent = m.group(1)
        body_indent = indent + "    "
        # 收集 try body（直到同缩进 except）
        j = i + 1
        body = []
        while j < len(lines):
            l = lines[j]
            if l.strip() == "":
                body.append(("blank", l))
                j += 1
                continue
            if l.startswith(indent) and not l.startswith(body_indent):
                break  # 提前缩出 → 无 except 的裸 try，跳过
            if re.match(r"^" + re.escape(indent) + r"except\b", l):
                break
            body.append(("code", l))
            j += 1
        # except + pass 校验
        if not (j < len(lines)
                and re.match(r"^" + re.escape(indent) + r"except\b", lines[j])
                and j + 1 < len(lines) and lines[j + 1].strip() == "pass"):
            out.append(line)
            i += 1
            skipped += 1
            continue
        # 语义校验：body 中任何处于语句首行的内容必须为 import 或 blog/diag 调用
        safe = True
        pending_call = False  # 上一个语句是跨行调用（允许续行）
        for tag, l in body:
            if tag == "blank":
                continue
            if l.startswith(body_indent) or l.startswith(body_indent.rstrip() + "\t"):
                stmt = l.strip()
                if IMPORT_RE.search(stmt):
                    pending_call = False
                    continue
                if re.match(r"^(blog|diag)\(", stmt):
                    pending_call = True
                    continue
                if re.match(r"^\w+\(.*\)$", stmt) and pending_call:
                    continue  # 保守：上一语句的结尾行（同行续行）
                safe = False
                break
            # 深缩进 = 续行（跨行参数）
            if pending_call:
                continue
            safe = False
            break
        if not safe:
            out.append(line)
            i += 1
            skipped += 1
            continue
        # 转换：无 try/except，import 换 safe_*，调用换 safe_*
        pend = False
        for tag, l in body:
            if tag == "blank":
                out.append(l)
                continue
            s = l
            for old, new in CALL_IMPORT_MAP.items():
                if old in s:
                    s = s.replace(old, new)
            s = re.sub(r"\bblog\(", "safe_blog(", s)
            s = re.sub(r"\bdiag\(", "safe_diag(", s)
            out.append(s)
        i = j + 2  # 跳过 except 与 pass
        converted += 1
    open(path, "w", encoding="utf-8").write("\n".join(out))
    print(f"{path}: converted={converted} skipped={skipped}")


for p in FILES:
    transform(p)
print("done")