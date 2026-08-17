# -*- coding: utf-8 -*-
"""v8.13 安全修复回归：路径逃逸（readfile / pdf_read / file_ops）+ 写大小上限。

背景：审计发现三类路径校验只覆盖"绝对路径"分支，相对路径 + .. 可逃逸到
项目目录外；FILE_WRITE_MAX_SIZE_MB 此前为死配置。本文件锁定修复后的行为。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_readfile_relative_escape():
    print("[SEC-1] read_local_file 相对路径逃逸拦截")
    from src.tools.readfile import read_local_file
    # v8.13 第四批: read_local_file 已改 sync 工具，.invoke 同步调用
    r = read_local_file.invoke({"path": "../../../outside_secret.txt"})
    check("相对路径逃逸被拒", "拒绝" in r or "ERR_PARSE" in r, r[:80])


def test_pdf_read_relative_escape():
    print("[SEC-2] pdf_read 相对路径逃逸拦截")
    from src.tools.search import pdf_read
    content, _ = pdf_read.func("../../../outside.pdf")
    check("pdf 相对路径逃逸被拒", "Access denied" in content, content[:80])


def test_file_ops_prefix_escape():
    print("[SEC-3] write_local_file 同前缀目录逃逸拦截（output_evil）")
    from src.tools.file_ops import write_local_file
    # output/../output_evil 解析后落在 workspace/output_evil/（不是 output/ 子目录）
    r = write_local_file.func("output/../output_evil/x.md", "x", "write")
    check("同前缀逃逸被拒", "Access denied" in r, r[:80])
    # 正常写入不受影响（放行 + 清理）
    r2 = write_local_file.func("sec_ok.md", "内容", "write")
    check("正常写入仍放行", r2.startswith("Success"), r2[:60])
    from src.core.write_pipeline import _WORKSPACE_ROOT
    (_WORKSPACE_ROOT / "sec_ok.md").unlink(missing_ok=True)


def test_write_size_limit():
    print("[SEC-4] FILE_WRITE_MAX_SIZE_MB 写大小上限生效（此前死配置）")
    from src.config import settings
    from src.tools.file_ops import write_local_file
    old = settings.FILE_WRITE_MAX_SIZE_MB
    settings.FILE_WRITE_MAX_SIZE_MB = 0.0001  # ≈100 字节，触发拒绝
    try:
        r = write_local_file.func("sec_big.md", "x" * 5000, "write")
        check("超限写入被拒", "超限" in r or "Error" in r, r[:80])
    finally:
        settings.FILE_WRITE_MAX_SIZE_MB = old


print()
if __name__ == "__main__":
    test_readfile_relative_escape()
    test_pdf_read_relative_escape()
    test_file_ops_prefix_escape()
    test_write_size_limit()
    print(f"security tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        raise SystemExit(1)
