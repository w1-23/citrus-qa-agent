# -*- coding: utf-8 -*-
"""v8.13-b5a AgentLoop 基座原语单元测试（tc_id / count_unique_docs / last_message_content /
invoke_llm_with_retry / force_final_answer）。"""
import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.agent_loop import (
    tc_id, count_unique_docs, last_message_content,
    invoke_llm_with_retry, force_final_answer, FINAL_ANSWER_PROMPT,
)


class _Call:
    def __init__(self, cid):
        self.id = cid


class _Resp:
    def __init__(self, content):
        self.content = content


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _noop_sleep(_s):
    pass


def test_tc_id():
    assert tc_id({"id": "a1"}) == "a1"
    assert tc_id({"name": "x"}) != ""      # dict 无 id → 生成 uuid
    assert tc_id(_Call("b2")) == "b2"
    assert tc_id(_Call("")) != ""          # 对象无 id → 生成 uuid


def test_count_unique_docs():
    rows = [
        {"doi": "10.1/AAA"},
        {"doi": "10.1/aaa"},         # DOI 大小写不敏感去重
        {"doi": "10.2/BBB"},
        {"title": "no doi"},          # 无 DOI 按条计数
    ]
    assert count_unique_docs(rows) == 3
    assert count_unique_docs([]) == 0


def test_last_message_content():
    msgs = [SystemMessage(content="sys"), HumanMessage(content="human"), AIMessage(content="ai")]
    assert last_message_content(msgs, "aimessage") == "ai"
    assert last_message_content(msgs, "any") == "ai"
    assert last_message_content(msgs, "nonsystem") == "ai"
    # aimessage 模式无 AIMessage → 空
    assert last_message_content([HumanMessage(content="h")], "aimessage") == ""
    # any 模式回退到任意含 content 消息（含 system）
    assert last_message_content([SystemMessage(content="sys")], "any") == "sys"
    # nonsystem 模式跳过 system，回退到 human
    assert last_message_content([SystemMessage(content="sys"), HumanMessage(content="h")], "nonsystem") == "h"


def test_invoke_llm_retry_success_first_try():
    async def ok():
        return _Resp("ok")

    async def main():
        return await invoke_llm_with_retry(ok, label="T")
    resp, attempts, err = _run(main())
    assert resp.content == "ok" and attempts == 1 and err == ""


def test_invoke_llm_retry_success_after_failure(monkeypatch):
    state = {"n": 0}

    async def flaky():
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("flaky")
        return _Resp("ok")

    monkeypatch.setattr("src.core.agent_loop.asyncio.sleep", _noop_sleep)
    async def main():
        return await invoke_llm_with_retry(flaky, label="T")
    resp, attempts, err = _run(main())
    assert resp.content == "ok" and attempts == 2 and err == "flaky"


def test_invoke_llm_retry_exhausted_none(monkeypatch):
    async def always_fail():
        raise RuntimeError("boom")

    monkeypatch.setattr("src.core.agent_loop.asyncio.sleep", _noop_sleep)
    async def main():
        return await invoke_llm_with_retry(always_fail, label="T", on_exhausted="none")
    resp, attempts, err = _run(main())
    assert resp is None and attempts == 3 and err == "boom"


def test_invoke_llm_retry_exhausted_raise(monkeypatch):
    async def always_fail():
        raise RuntimeError("boom")

    monkeypatch.setattr("src.core.agent_loop.asyncio.sleep", _noop_sleep)
    async def main():
        await invoke_llm_with_retry(always_fail, label="T", on_exhausted="raise")
    with pytest.raises(RuntimeError):
        _run(main())


def test_force_final_success():
    async def sc():
        return _Resp("final content")
    msgs = [HumanMessage(content="h"), AIMessage(content="prev")]
    assert _run(force_final_answer(msgs, stream_call=sc)) == "final content"


def test_force_final_fallback_on_error():
    async def sc():
        raise RuntimeError("boom")
    msgs = [HumanMessage(content="h"), AIMessage(content="prev answer")]
    assert _run(force_final_answer(msgs, stream_call=sc, label="[T]", fallback_mode="aimessage")) == "prev answer"


def test_force_final_empty_content_fallback():
    async def sc():
        return _Resp("")
    msgs = [HumanMessage(content="h"), AIMessage(content="prev")]
    assert _run(force_final_answer(msgs, stream_call=sc, fallback_mode="aimessage")) == "prev"


def test_final_prompt_non_empty():
    assert "不要提及工具或轮次限制" in FINAL_ANSWER_PROMPT