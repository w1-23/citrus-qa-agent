"""Safe declarative SVG drawing tool exposed only when visualization is enabled."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from langchain_core.tools import tool

_NS = "http://www.w3.org/2000/svg"
_TAGS = {"svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline",
         "polygon", "text", "tspan", "title", "desc", "defs", "linearGradient",
         "radialGradient", "stop", "clipPath", "marker"}
_ATTRS = {"viewBox", "preserveAspectRatio", "stroke-width", "stroke-linecap",
          "stroke-linejoin", "stroke-dasharray", "stroke-dashoffset", "fill-opacity",
          "stroke-opacity", "font-family", "font-size", "font-weight", "text-anchor",
          "dominant-baseline", "gradientUnits", "gradientTransform", "stop-color",
          "stop-opacity", "clipPathUnits", "markerWidth", "markerHeight", "markerUnits",
          "marker-start", "marker-mid", "marker-end", "aria-label", "clip-path", "role",
          "id", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
          "width", "height", "d", "points", "fill", "stroke", "opacity", "transform",
          "offset"}


def _sanitize_svg(source: str) -> str:
    source = str(source or "").strip()
    if not source or len(source) > 100_000 or re.search(r"<!DOCTYPE|<!ENTITY", source, re.I):
        raise ValueError("SVG is empty, too large, or contains a forbidden declaration")
    root = ET.fromstring(source)
    if root.tag != f"{{{_NS}}}svg":
        raise ValueError("Root element must be an SVG document")
    nodes = 0

    def clean(node: ET.Element) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 5000 or node.tag.startswith("{") and node.tag.split("}", 1)[0][1:] != _NS:
            raise ValueError("Unsupported SVG namespace or complexity")
        tag = node.tag.split("}", 1)[-1]
        if tag not in _TAGS:
            raise ValueError(f"Unsupported SVG element: {tag}")
        node.tag = tag
        for key in list(node.attrib):
            value = node.attrib[key]
            if key not in _ATTRS or len(value) > 4000 or re.search(r"[<>\x00-\x1f]", value):
                del node.attrib[key]
                continue
            if key == "id" and not re.fullmatch(r"[A-Za-z][\w:.-]{0,100}", value):
                del node.attrib[key]
                continue
            if re.search(r"url\s*\(", value, re.I) and not re.fullmatch(
                    r"url\(\s*#[A-Za-z][\w:.-]{0,100}\s*\)", value, re.I):
                del node.attrib[key]
                continue
            if re.search(r"javascript:|data:|https?:|file:", value, re.I):
                del node.attrib[key]
        for child in list(node):
            clean(child)

    clean(root)
    if not root.get("viewBox") and not root.get("width"):
        raise ValueError("SVG must provide viewBox or width")
    root.set("xmlns", _NS)
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)


@tool
def draw_svg(svg: str = "", alt_text: str = "图形") -> str:
    """Validate and return an SVG diagram for the final answer. Use only when visualization is enabled and a diagram materially helps."""
    source = str(svg or "").strip()
    fenced = re.fullmatch(r"```(?:svg|xml)?\s*([\s\S]*?)\s*```", source, re.I)
    if fenced:
        source = fenced.group(1).strip()
    else:
        start = source.lower().find("<svg")
        end = source.lower().rfind("</svg>")
        if start >= 0 and end >= start:
            source = source[start:end + len("</svg>")]
    if not source:
        return "[SVG_INPUT_MISSING] 请提供完整的 SVG 文档；需要以 <svg> 开始并以 </svg> 结束。"
    try:
        safe = _sanitize_svg(source)
    except (ET.ParseError, ValueError) as exc:
        return f"[SVG_FORMAT_ERROR] SVG 未通过安全校验：{exc}。请修正结构后重试。"
    label = re.sub(r"[\r\n<>]", " ", str(alt_text or "图形")).strip()[:160] or "图形"
    return f"SVG 已通过安全校验。请在最终回答中按原样嵌入以下可视化代码块（可加简短说明）：\n\n```svg\n{safe}\n```\n\n图示说明：{label}"
