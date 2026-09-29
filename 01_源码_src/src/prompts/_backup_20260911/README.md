# 提示词旧版备份（2026-09-11）

## 备份原因
用户决策（v10 输出规范精简）：
- 原 09~15 共 7 份输出格式模板（事实/机制/对比/综述/实验/任务/默认）引导过细、约束过多，
  改为「按用户原始问题自主组织 + 少量硬约束」。
- supervisor 提示词中原 v9.5 规则 7（结论一致性自检）与规则 8（时效触发）迁往代码层
  （src/core/audit.py 架构级审计），不再靠提示词约束。

## 恢复方法
- 源文件：把 source\*.md 拷回 ../source/
- 构建产物：把 builds\*.md 拷回 ../builds/（或改完 source 后运行
  `python -c "from src.prompts.loader import ensure_fixed_prompts as e; e(reload=True)"` 重建）
- 快照：把 snapshots\* 拷回 ../snapshots/（或 `python -m src.prompts.snapshot` 重新渲染）

## 内容
- source/：v9.6 源文件全量（21 份）
- builds/：v9.6 固定拼接产物（6 份角色提示词）
- snapshots/：v9.6 快照（角色 + source 镜像）
