# -*- coding: utf-8 -*-
"""pytest 全局配置（v8.4.3 工单10 + v8.4.14 沙箱规避）。

测试进程把日志重定向到工作区内临时目录（tests/.tmp_runner/），避免
测试帧混入生产 logs/agent.log，也避免 DSH 沙箱下系统 TEMP 不可写导致的
环境性失败。必须在任何 src 模块导入前设置（本文件在测试收集前被 pytest 导入）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _tmpenv import tmp_dir  # noqa: E402  （tests 目录已在 sys.path）

os.environ["CITRUS_LOG_DIR"] = str(tmp_dir("logs"))
