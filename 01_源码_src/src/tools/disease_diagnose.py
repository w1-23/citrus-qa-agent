"""柑橘病害图像初筛工具（仅在用户本轮上传了图片时暴露给模型）。"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.vision import labels as label_bank
from src.vision.context import get_request_images
from src.vision.engine import get_engine

logger = logging.getLogger(__name__)


@tool
def disease_diagnose(image_index: int = 0, focus: str = "") -> str:
    """Screen citrus disease symptoms in an image uploaded in this turn.

    The result is a compact screening signal from fixed-class pretrained models.
    A classifier score is not diagnostic confidence. ``focus`` optionally names
    the symptom the user cares about.
    """
    images = get_request_images()
    if not images:
        return ("[ERR_NO_IMAGE] 本轮没有可用的图片附件，无法做图像识别。"
                "如果用户只是文字提问，请按知识库/联网证据回答，不要编造图像结论。")
    try:
        index = int(image_index)
    except (TypeError, ValueError):
        index = 0
    if index < 0 or index >= len(images):
        names = "；".join(f"{i}. {img.name}" for i, img in enumerate(images))
        return (f"[ERR_IMAGE_INDEX] image_index={index} 超出范围；本轮共 {len(images)} 张图片："
                f"{names}。请用 0..{len(images) - 1} 之间的序号重试。")

    image = images[index]
    diag = get_engine().diagnose_data_url(image.url)
    report = _compact_screening_result(diag)
    if focus and str(focus).strip():
        report += f"\n用户关注：{str(focus).strip()[:80]}。"
    logger.info("[Vision] disease_diagnose index=%d status=%s top=%s runs=%d views=%d "
                "stage2=%s providers=%s elapsed_ms=%d quality=%s",
                index, diag.status, diag.top[:1], diag.inference_runs, diag.views,
                diag.stage2_triggered, diag.providers, diag.elapsed_ms,
                diag.quality_issues)
    if diag.status != "ok":
        report += ("\n【需要主模型兜底】本次分类器未达到可靠筛查条件。若主模型支持图像输入，必须重新检查本轮原图，"
                   "不受分类器候选限制，独立给出最可能判断和定性把握；若主模型不支持图像输入，必须说明无法复核原图，"
                   "并仅根据现有线索谨慎回答或请求补充清晰照片/症状。")
    report += ("\n给主模型：最终回答只保留“最可能判断及把握、1 个关键鉴别点、必要时 1 条筛查建议”；"
               "分类器与可见症状冲突时以原图证据为先；不展开候选清单、类别目录、机制或多项栽培建议。"
               "分类器分数不是诊断概率；这是初筛，不是诊断结论，不得写成确诊或据此推荐农药用量。")
    return report


def _compact_screening_result(diag) -> str:
    """Keep only screening signals the main model needs for a short user answer."""
    if diag.status == "error":
        return f"病害图像初筛不可用：{diag.error or '图片读取或推理失败'}。"
    if diag.status == "no_model":
        return "病害图像初筛不可用：本机没有可运行的识别模型。"
    if diag.status == "low_quality":
        issue = "、".join(str(item) for item in diag.quality_issues[:2])
        return "病害图像初筛不可用：照片质量不足" + (f"（{issue}）" if issue else "") + "。"

    if diag.top:
        key, score = diag.top[0]
        result = (f"病害图像初筛状态：{'通过筛查阈值' if diag.status == 'ok' else '未达可信阈值'}；"
                  f"最高候选为“{label_bank.zh_name(key)}”，分类器分数 {score:.0%}。")
        if label_bank.is_weak(key):
            result += "此类别为弱参考标签，证据更弱。"
    else:
        result = "病害图像初筛状态：没有可用的病名候选。"
    result += ("此分类器只覆盖有限病害类别；着色异常、缺素、药害和未覆盖病害可能不在范围内，"
               "未命中或低分不等于排除。分类器分数不是诊断概率。")
    return result
