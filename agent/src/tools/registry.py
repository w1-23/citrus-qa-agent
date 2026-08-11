"""Tool Registry — tool classification, partitioned concurrency, offload"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any, Optional
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from src.config import settings

logger = logging.getLogger(__name__)

HITL_DANGEROUS_TOOLS = set()
AUTO_OFFLOAD_THRESHOLD = 15000
OFFLOAD_DIR = Path(settings.WORKSPACE_DIR) / "tmp"

_offload_created_files: list[Path] = []


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
        return f"[ERR_HITL_REJECT] 权限不足: {e}。建议: 检查文件权限或改用工作区内路径"
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


async def _check_tool_sandbox(tool_name: str, args: dict) -> Optional[str]:
    try:
        from src.engine.sandbox import check_sandbox
        decision = await check_sandbox(tool_name, args)
        if not decision.allowed:
            return f"[ERR_HITL_REJECT] 沙箱拒绝: {decision.reason}"
        if decision.requires_approval:
            return None
    except ImportError:
        pass  # sandbox module not available — allow all tools
    return None


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

    if response_format == "content_and_artifact" and raw_func is not None:
        try:
            content, artifact = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None, lambda: raw_func(**args)
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
            result = await asyncio.wait_for(tool.ainvoke(args), timeout=exec_timeout)
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
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
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
