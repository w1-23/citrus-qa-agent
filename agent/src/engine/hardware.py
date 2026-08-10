import logging
import onnxruntime as ort
from src.config import _yaml_val

logger = logging.getLogger(__name__)

def _get_provider_priority() -> list[str]:
    yaml_priorities = _yaml_val("onnx", "provider_priority", default=None)
    if yaml_priorities and isinstance(yaml_priorities, list):
        return yaml_priorities
    return [
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

def get_ort_providers() -> list[str]:
    raw_providers = ort.get_available_providers()
    logger.info(f"[Hardware] ONNX 原始可用 Providers: {raw_providers}")

    priority = _get_provider_priority()
    available = set(raw_providers)
    matched = [p for p in priority if p in available]
    if not matched:
        matched = ["CPUExecutionProvider"]

    active_gpu = [p for p in matched if p != "CPUExecutionProvider"]
    if active_gpu:
        logger.info(f"[Hardware] GPU 加速已激活 | Provider: {active_gpu[0]} | 优先级: {priority}")
    else:
        logger.warning(f"[Hardware] 未检测到 GPU Provider，降级为 CPUExecutionProvider")

    return matched