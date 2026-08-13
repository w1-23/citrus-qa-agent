# -*- coding: utf-8 -*-
"""pytest 全局配置（v8.4.3 工单10: 测试日志独立 sink）。

测试进程把日志重定向到临时目录，避免测试帧混入生产 logs/agent.log。
必须在任何 src 模块导入前设置（本文件在测试收集前被 pytest 导入）。
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_tmp_log = Path(tempfile.gettempdir()) / "citrus_test_logs"
_tmp_log.mkdir(parents=True, exist_ok=True)
os.environ["CITRUS_LOG_DIR"] = str(_tmp_log)
