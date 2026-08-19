"""Tool Registry — tool classification, partitioned concurrency, offload"""
import asyncio
import contextvars
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Optional
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)

AUTO_OFFLOAD_THRESHOLD = 15000
# v8.13: 锚定 PROJECT_ROOT（此前 CWD 相对——从非仓库根启动时 offload 写入错误位置）
OFFLOAD_DIR = (PROJECT_ROOT / settings.WORKSPACE_DIR / "tmp").resolve()

_offload_created_files: list[Path] = []

# v8.4.5: ask 模式授权等待——权限拒绝时工具执行挂起等待前端卡片授权，
# 授权到达后在同一执行内继续（不再要求用户整轮重跑）。
_pending_permission_events: dict = {}


def signal_permission_granted(session_id: str, tool_name: str) -> None:
    """授权到达后唤醒对应挂起事件（由 /api/v2/permission/grant 端点调用）。"""
    for key in (f"{session_id}|{tool_name}", f"{session_id}|*"):
        ev = _pending_permission_events.pop(key, None)
        if ev is not None:
            try:
                ev.set()
            except Exception:
                pass


@dataclass
class ToolSpec:
    func: Callable
    category: str = "general"
    is_readonly: bool = True
    concurrency_key: Optional[str] = None


_TOOL_SPECS: dict[str, ToolSpec] = {}


def register_tool(tool: BaseTool, category: str = "general", is_readonly: bool = True, concurrency_key: Optional[str] = None):
    _TOOL_SPECS[tool.name] = ToolSpec(func=tool, category=category, is_readonly=is_readonly, concurrency_key=concurrency_key)


def get_tool_spec(name: str) -> Optional[ToolSpec]:
    return _TOOL_SPECS.get(name)


def get_tools_by_category(category: str) -> list[str]:
    return [name for name, spec in _TOOL_SPECS.items() if spec.category == category]


def _classify_error(e: Exception) -> str:
    """统一错误回传 (v8.3.1): [ERR_类别] 原因 + 建议策略，让 LLM 对症下药。"""
    if isinstance(e, ImportError):
        pkg = getattr(e, "name", str(e))
        return f"[ERR_MISSING_DEP] 缺少依赖库: {pkg}。建议: pip install {pkg} 后重试"
    if isinstance(e, FileNotFoundError):
        return f"[ERR_FILE_NOT_FOUND] 文件不存在: {e.filename}。建议: 检查路径拼写与文件位置"
    if isinstance(e, PermissionError):
        # v8.4.5: 操作系统权限错误与 HITL 拒绝区分（[ERR_PERMISSION] vs [ERR_HITL_REJECT]）
        return f"[ERR_PERMISSION] 权限不足: {e}。建议: 检查文件权限或改用工作区内路径"
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return f"[ERR_NETWORK] 请求超时: {e}。建议: 稍后重试或改用本地源"
    msg = str(e)
    if "timeout" in msg.lower() or "connection" in msg.lower():
        return f"[ERR_NETWORK] 网络错误: {e}。建议: 稍后重试或改用本地源"
    if "parse" in msg.lower() or "decode" in msg.lower() or "json" in msg.lower():
        return f"[ERR_PARSE] 数据解析错误: {e}。建议: 检查输入参数格式"
    return f"[ERR_PARSE] 工具执行异常: {e}。建议: 检查参数是否合理"


def _offload_large_result(result: str, tool_name: str) -> str:
    if len(result) <= AUTO_OFFLOAD_THRESHOLD:
        return result
    try:
        offload_dir = OFFLOAD_DIR
        offload_dir.mkdir(parents=True, exist_ok=True)
        file_id = uuid4().hex[:8]
        path = offload_dir / f"{tool_name}_{file_id}.txt"
        path.write_text(result, encoding="utf-8")
        _offload_created_files.append(path)
        logger.info(f"[Offload] {tool_name} 返回 {len(result)} 字符 → 已卸载到 {path}")
        rel_path = Path("tmp") / path.name
        return (
            f"[结果过长({len(result)}字符)，已自动卸载]\n"
            f"路径: {rel_path.as_posix()}\n"
            f"摘要: {result[:500]}...\n"
            f"如需完整内容请调用 read_local_file('{rel_path.as_posix()}')"
        )
    except Exception as e:
        logger.warning(f"[Offload] 卸载失败: {e}")
        return result


def get_offload_file_list() -> list[Path]:
    return list(_offload_created_files)


def cleanup_offload_files() -> int:
    count = 0
    for p in _offload_created_files:
        try:
            if p.exists():
                p.unlink()
                count += 1
        except Exception as e:
            logger.warning(f"[Cleanup] 删除 {p} 失败: {e}")
    _offload_created_files.clear()
    if count > 0:
        logger.info(f"[Cleanup] 已清理 {count} 个 offload 临时文件")
    return count


# v8.4.3: 只读/分析类工具白名单（注册表未初始化时替代 spec 放行，
# 与 config.yaml tools.registrations 的 readonly 语义一致）
_READONLY_TOOLS = frozenset({
    "citrus_rag_search", "academic_search", "pdf_read",
    "read_local_file", "statistical_analysis", "experimental_design",
    "fetch_fulltext",
})


def _is_workspace_output_path(path: str) -> bool:
    """write_local_file 路径是否落在 workspace/output 内（auto_workspace 免询问依据）。

    v8.13: 收敛到 core.path_policy.is_output_path（统一 is_relative_to，原
    startswith 同前缀漏洞 registry 侧残留一并消除）。
    """
    from src.core.path_policy import is_output_path
    return is_output_path(path)


async def _check_tool_sandbox(tool_name: str, args: dict) -> Optional[str]:
    """沙箱/权限检查（v8.3.3 fail-closed 内置 + v8.4.3 结构化权限）。

    策略:
      - SANDBOX_ENABLED=False → 放行
      - 只读/分析类工具 → 放行（检索/读取/统计类）
      - 危险模式（delete_*/exec_*/write_*）→ 按 permission_mode:
          auto_workspace: write_local_file 且路径在 workspace/output 内 → 放行
          ask: 查 permission_grants（workspace/once/session 范围）→ 放行或
              拒绝并发 permission_request 事件（前端审批卡片）
          deny: 一律拒绝
      - 检查异常 → 拒绝执行（fail-closed）
    """
    try:
        if not getattr(settings, "SANDBOX_ENABLED", True):
            return None
        import fnmatch
        spec = get_tool_spec(tool_name)
        if spec is not None and (spec.is_readonly or spec.category in ("analysis", "agent", "file")):
            return None
        if tool_name in _READONLY_TOOLS:
            return None
        dangerous = getattr(settings, "SANDBOX_DANGEROUS_PATTERNS", None) or []
        if not any(fnmatch.fnmatch(tool_name, p) for p in dangerous):
            return None

        mode = getattr(settings, "PERMISSION_MODE", "auto_workspace") or "auto_workspace"
        path = str((args or {}).get("path", "") or "")

        if mode == "auto_workspace":
            if tool_name == "write_local_file" and _is_workspace_output_path(path):
                return None
            return (f"[ERR_HITL_REJECT] 沙箱拒绝: 工具 {tool_name} 属危险操作类，"
                    f"仅允许写入 workspace/output（permission_mode=auto_workspace）")

        if mode == "ask":
            try:
                from src.session.manager import session_manager
                from src.core.tracing import get_session_id
                sid = get_session_id()
                granted = await asyncio.to_thread(
                    session_manager.consume_grant, sid, tool_name, path)
                if granted:
                    return None
            except Exception:
                pass
            # 发结构化权限请求事件 → 前端审批卡片
            try:
                from src.core.progress_bus import emit_encoded
                emit_encoded("permission_request", {
                    "tool_name": tool_name,
                    "args": {k: str(v)[:200] for k, v in (args or {}).items()},
                })
            except Exception:
                pass
            # v8.4.5: 挂起等待授权（授权后同一执行内继续；超时按拒绝处理）
            try:
                from src.core.tracing import get_session_id as _get_sid
                sid = _get_sid()
                wait_key = f"{sid}|{tool_name}"
                ev = asyncio.Event()
                _pending_permission_events[wait_key] = ev
                try:
                    await asyncio.wait_for(ev.wait(), timeout=settings.PERMISSION_WAIT_SEC)
                    granted = await asyncio.to_thread(
                        session_manager.consume_grant, sid, tool_name, path)
                    if granted:
                        logger.info(f"[Sandbox] 权限已授权，同执行内继续: tool={tool_name}")
                        return None
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[Sandbox] 权限等待超时({settings.PERMISSION_WAIT_SEC}s): {tool_name}")
                finally:
                    _pending_permission_events.pop(wait_key, None)
            except Exception:
                pass
            return (f"[ERR_HITL_REJECT] 权限未授权: 工具 {tool_name} "
                    f"path={path[:120] or '(无)'}（permission_mode=ask，"
                    f"请在弹出的审批卡片中点击允许）")

        # deny 或未知模式
        return (f"[ERR_HITL_REJECT] 沙箱拒绝: 工具 {tool_name} 属危险操作类，"
                f"未获执行批准（permission_mode={mode}）")
    except Exception as e:
        logger.error(f"[Sandbox] 检查异常(fail-closed): {e}")
        return f"[ERR_HITL_REJECT] 沙箱检查失败，已拒绝执行: {e}"


async def _invoke_tool_with_ctx(ctx: contextvars.Context, tool: BaseTool, args: dict):
    """工具执行时保持请求 contextvar（v8.3.4）。

    run_in_executor 线程不继承 contextvars → 工具内部的 emit_*（tool_progress 等）
    会落到全局兜底队列而丢失（执行卡片缺失的根因）。
    async 工具协程天然继承 contextvar，直接 await；sync 工具用 ctx.run 注入。
    """
    if getattr(tool, "coroutine", None) is not None:
        return await tool.ainvoke(args)
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: ctx.run(lambda: tool.invoke(args)))


async def _run_single_tool(tool: BaseTool, tool_call: dict) -> ToolMessage:
    tool_name = tool_call.get("name", tool.name)
    args = tool_call.get("args", {})

    reject_reason = await _check_tool_sandbox(tool_name, args)
    if reject_reason:
        return ToolMessage(content=reject_reason, tool_call_id=tool_call["id"], name=tool_name, artifact={})

    exec_timeout = getattr(settings, "TOOL_EXEC_TIMEOUT_SEC", 60) or 60

    # 注意：对于 response_format="content_and_artifact" 的工具，tool.ainvoke()
    # 会丢弃 artifact 只返回 content 字符串。
    # 必须直接调用底层函数才能获取完整的 (content, artifact) 元组。
    response_format = getattr(tool, "response_format", None)
    raw_func = getattr(tool, "func", None)
    ctx = contextvars.copy_context()

    if response_format == "content_and_artifact" and raw_func is not None:
        try:
            content, artifact = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, lambda: ctx.run(lambda: raw_func(**args))
                ),
                timeout=exec_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"[PartitionedNode] 工具 {tool.name} 执行超时(>{exec_timeout}s)")
            return ToolMessage(
                content=f"[ERR_TIMEOUT] 工具 {tool.name} 执行超过 {exec_timeout}s，已中断。可稍后重试或换更小范围。",
                tool_call_id=tool_call["id"], name=tool.name, artifact={})
        except Exception as e:
            logger.error(f"[PartitionedNode] 工具 {tool.name} 执行失败: {e}")
            error_msg = _classify_error(e)
            return ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name=tool.name, artifact={})
        if isinstance(content, str):
            content = _offload_large_result(content, tool_name)
        return ToolMessage(content=content, artifact=artifact, tool_call_id=tool_call["id"], name=tool_name)
    else:
        try:
            result = await asyncio.wait_for(_invoke_tool_with_ctx(ctx, tool, args),
                                            timeout=exec_timeout)
        except asyncio.TimeoutError:
            logger.error(f"[PartitionedNode] 工具 {tool.name} 执行超时(>{exec_timeout}s)")
            return ToolMessage(
                content=f"[ERR_TIMEOUT] 工具 {tool.name} 执行超过 {exec_timeout}s，已中断。可稍后重试或换更小范围。",
                tool_call_id=tool_call["id"], name=tool.name, artifact={})
        except Exception as e:
            logger.error(f"[PartitionedNode] 工具 {tool.name} 执行失败: {e}")
            error_msg = _classify_error(e)
            return ToolMessage(content=error_msg, tool_call_id=tool_call["id"], name=tool.name, artifact={})

        if not isinstance(result, ToolMessage):
            if isinstance(result, str):
                result = ToolMessage(content=_offload_large_result(result, tool_name), tool_call_id=tool_call["id"], name=tool_name)
            elif isinstance(result, tuple) and len(result) == 2:
                content, artifact = result
                if isinstance(content, str):
                    content = _offload_large_result(content, tool_name)
                result = ToolMessage(content=content, artifact=artifact, tool_call_id=tool_call["id"], name=tool_name)
            else:
                result = ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)
        elif isinstance(result.content, str):
            result.content = _offload_large_result(result.content, tool_name)

        return result


async def run_tool_checked(tool: BaseTool, args: dict) -> str:
    """v8.13 第四批: 工具执行统一出口——供图内联/流水线直调复用。

    此前 write_pipeline 直调 write_local_file.func、expert_graph 内联
    read/pdf/write 绕过 _check_tool_sandbox 与超时/offload（ask 审批模式对
    这些路径形同虚设，且大文件阻塞事件循环）。统一经此出口：
    - 沙箱/权限检查（fail-closed）
    - 超时中断（TOOL_EXEC_TIMEOUT_SEC）
    - 大结果 offload + 错误分类
    Returns: content 字符串。以 [ERR_* 或 "Error:" 前缀判失败。
    """
    tool_name = tool.name
    reject_reason = await _check_tool_sandbox(tool_name, args)
    if reject_reason:
        return reject_reason

    exec_timeout = getattr(settings, "TOOL_EXEC_TIMEOUT_SEC", 60) or 60
    ctx = contextvars.copy_context()
    raw_func = getattr(tool, "func", None)
    response_format = getattr(tool, "response_format", None)

    try:
        if response_format == "content_and_artifact" and raw_func is not None:
            content, _artifact = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, lambda: ctx.run(lambda: raw_func(**args))),
                timeout=exec_timeout)
            return content if isinstance(content, str) else str(content)
        result = await asyncio.wait_for(_invoke_tool_with_ctx(ctx, tool, args),
                                        timeout=exec_timeout)
        if isinstance(result, ToolMessage):
            content = result.content
        elif isinstance(result, str):
            content = result
        elif isinstance(result, tuple) and len(result) == 2:
            content = result[0]
        else:
            content = str(result)
        return content if isinstance(content, str) else str(content)
    except asyncio.TimeoutError:
        return f"[ERR_TIMEOUT] 工具 {tool_name} 执行超过 {exec_timeout}s，已中断。"
    except Exception as e:
        return _classify_error(e)


class PartitionedToolNode:
    """Partitioned concurrency tool node — LangGraph compatible"""

    def __init__(self, tools: list[BaseTool]):
        self.tools_by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    async def _execute_group(self, group: list[dict]) -> list[ToolMessage]:
        results = []
        for tc in group:
            tool = self.tools_by_name.get(tc.get("name", ""))
            if tool is None:
                logger.warning(f"[PartitionedNode] 未知工具: {tc.get('name')}")
                results.append(ToolMessage(
                    content=f"[ERR_UNKNOWN_TOOL] 工具 '{tc.get('name')}' 不存在。可用工具: {list(self.tools_by_name.keys())}",
                    tool_call_id=tc.get("id", "unknown"),
                    name=tc.get("name", "unknown"),
                    artifact={},
                ))
                continue
            t_tool = time.perf_counter()
            msg = await _run_single_tool(tool, tc)
            dt_tool = (time.perf_counter() - t_tool) * 1000
            logger.info(f"[PartitionedNode] tool {tool.name} done ({dt_tool:.0f}ms)")
            # v8.4.6 B8: 结构化结果 envelope——熔断/统计只读 _meta 字段，
            # 不解析自由文本（书 §1.2.2"验证：输入隔离"）
            try:
                _content = str(getattr(msg, "content", "") or "")
                _meta = {"tool": tool.name, "status": "ok", "code": "OK"}
                if _content.startswith("[DEDUP]"):
                    _meta.update(status="skipped", code="DEDUP")
                elif _content.startswith("[ERR") or _content.startswith("[Error"):
                    _m = re.match(r"\[([A-Z_]+)\]", _content)
                    _meta.update(status="error",
                                 code=_m.group(1) if _m else "ERR_UNKNOWN")
                _art = getattr(msg, "artifact", None) or {}
                if isinstance(_art, dict):
                    _art = dict(_art)
                else:
                    _art = {}
                _art["_meta"] = _meta
                msg.artifact = _art
            except Exception:
                pass
            # v8.4.1: 业务日志——工具级事件（成功/失败/结果长度，排查检索与工具问题）
            try:
                from src.core.business_logger import blog
                content = str(getattr(msg, "content", ""))
                blog("tool_done", name=tool.name,
                     ms=int(dt_tool),
                     err=content.startswith("[ERR") or content.startswith("[Error"),
                     chars=len(content))
            except Exception:
                pass
            results.append(msg)
        return results

    async def _execute(self, tool_calls: list[dict]) -> list[ToolMessage]:
        if not tool_calls:
            return []
        groups: dict[str, list[dict]] = {}
        for tc in tool_calls:
            name = tc.get("name", "")
            spec = get_tool_spec(name)
            key = spec.concurrency_key if spec else name
            groups.setdefault(key, []).append(tc)
        logger.info(f"[PartitionedNode] {len(tool_calls)} 个工具调用, {len(groups)} 个并发组: {list(groups.keys())}")
        t_groups = time.perf_counter()
        tasks = [self._execute_group(group) for group in groups.values()]
        group_results = await asyncio.gather(*tasks)
        dt_groups = (time.perf_counter() - t_groups) * 1000
        all_results = []
        for grp in group_results:
            all_results.extend(grp)
        logger.info(f"[PartitionedNode] all {len(tool_calls)} tools done ({dt_groups:.0f}ms)")
        return all_results

    async def execute_tools(self, tool_calls: list[dict]) -> list[ToolMessage]:
        return await self._execute(tool_calls)

    async def ainvoke(self, inputs: dict) -> dict:
        messages = inputs.get("messages", [])
        if not messages:
            return {"messages": []}
        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return {"messages": []}
        results = await self._execute(tool_calls)
        return {"messages": results}

    def __call__(self, inputs: dict) -> dict:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        return loop.run_until_complete(self.ainvoke(inputs))


def init_tool_registry():
    from src.tools import _TOOL_REGISTRY_BY_NAME, _TOOL_REGISTRY
    import yaml
    from src.config import PROJECT_ROOT
    cfg_path = PROJECT_ROOT / "config.yaml"
    try:
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            registrations = cfg.get("tools", {}).get("registrations", [])
            if registrations:
                for reg in registrations:
                    name = reg.get("name")
                    tool = _TOOL_REGISTRY_BY_NAME.get(name)
                    if tool:
                        register_tool(tool, category=reg.get("category", "general"),
                                      is_readonly=reg.get("readonly", True),
                                      concurrency_key=reg.get("concurrency_key"))
                    else:
                        logger.warning(f"[ToolRegistry] 配置中工具 '{name}' 未找到，跳过")
                logger.info(f"[ToolRegistry] 从 YAML 注册 {len(registrations)} 个工具")
                return
    except Exception as e:
        logger.warning(f"[ToolRegistry] YAML 加载失败: {e}，使用硬编码 fallback")

    for t in _TOOL_REGISTRY:
        register_tool(t)
