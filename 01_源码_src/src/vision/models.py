"""ONNX 模型规格与会话缓存。

三个模型都是**公开预训练权重直接导出**，本仓库不训练、不微调：
- convnext: ixrbhii/citrus-disease-convnext —— 柑橘叶片 8 类（Apache-2.0）
- dima:     dima806/orange_fruit_disease_detection —— 橙果皮 3 类（Apache-2.0，模型卡已标注）
- mnv2:     linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification
            —— PlantVillage 38 类，只取黄龙病与白粉病弱参考类（license=other）

会话按 provider 链只建一次；缺模型文件时对应模型返回 ``None``，引擎整体降级而不是报错。
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, tuple[str, ...]], object] = {}
_LAST_PROVIDER: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    arch: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    labels: tuple[str, ...]
    mapping: dict[int, str] = field(default_factory=dict)
    weight: float = 1.0
    source: str = ""
    license: str = "unknown"
    #: 是否有可核验的标准许可登记；other/unknown 需由部署者明确启用
    licensed: bool = True

    def label_of(self, index: int) -> str | None:
        return self.mapping.get(int(index))


SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="convnext",
        filename="citrus_convnext.onnx",
        arch="convnext_small",
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        labels=("Bacterial Blight", "Black Spot", "Canker", "Dry Leaf",
                "Healthy", "Leaf Curl Virus", "Sooty Mould", "Spider Mites"),
        mapping={0: "bacterial_blight", 1: "black_spot", 2: "canker", 3: "dry_leaf",
                 4: "healthy", 5: "leaf_curl_virus", 6: "sooty_mould", 7: "spider_mites"},
        weight=1.0,
        source="ixrbhii/citrus-disease-convnext",
        license="apache-2.0",
    ),
    ModelSpec(
        key="dima",
        filename="dima_orange_vit.onnx",
        arch="vit_base_patch16_224",
        input_size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        labels=("citrus canker", "healthy", "melanose"),
        mapping={0: "canker", 1: "healthy", 2: "melanose"},
        weight=1.0,
        source="dima806/orange_fruit_disease_detection",
        license="apache-2.0",
        licensed=True,
    ),
    ModelSpec(
        key="plantvillage",
        filename="plantvillage_mnv2.onnx",
        arch="mobilenet_v2",
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        labels=(
            "Apple Scab", "Apple with Black Rot", "Cedar Apple Rust", "Healthy Apple",
            "Healthy Blueberry Plant", "Cherry with Powdery Mildew", "Healthy Cherry Plant",
            "Corn (Maize) with Cercospora and Gray Leaf Spot", "Corn (Maize) with Common Rust",
            "Corn (Maize) with Northern Leaf Blight", "Healthy Corn (Maize) Plant",
            "Grape with Black Rot", "Grape with Esca (Black Measles)",
            "Grape with Isariopsis Leaf Spot", "Healthy Grape Plant",
            "Orange with Citrus Greening", "Peach with Bacterial Spot", "Healthy Peach Plant",
            "Bell Pepper with Bacterial Spot", "Healthy Bell Pepper Plant",
            "Potato with Early Blight", "Potato with Late Blight", "Healthy Potato Plant",
            "Healthy Raspberry Plant", "Healthy Soybean Plant", "Squash with Powdery Mildew",
            "Strawberry with Leaf Scorch", "Healthy Strawberry Plant",
            "Tomato with Bacterial Spot", "Tomato with Early Blight", "Tomato with Late Blight",
            "Tomato with Leaf Mold", "Tomato with Septoria Leaf Spot",
            "Tomato with Spider Mites or Two-spotted Spider Mite", "Tomato with Target Spot",
            "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Healthy Tomato Plant",
        ),
        # 只保留黄龙病与白粉病弱参考类；其余类别不映射到柑橘病害。
        mapping={15: "greening", 25: "powdery_mildew"},
        weight=0.35,
        source="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
        license="other",
        licensed=False,
    ),
)


def enabled_specs(allow_unlicensed: bool = False) -> tuple[ModelSpec, ...]:
    """按许可策略挑出要加载的模型（unlicensed 需要显式开关）。"""
    return tuple(spec for spec in SPECS if spec.licensed or allow_unlicensed)


def default_model_dir() -> Path:
    """模型目录：环境变量 > config.yaml vision.model_dir > <PROJECT_ROOT>/models/vision。"""
    env = str(os.getenv("VISION_MODEL_DIR") or "").strip()
    if env:
        return Path(env).expanduser()
    try:
        from src.config import PROJECT_ROOT, settings
        configured = str(getattr(settings, "VISION_MODEL_DIR", "") or "").strip()
        if configured:
            return Path(configured).expanduser()
        return Path(PROJECT_ROOT) / "models" / "vision"
    except Exception:  # noqa: BLE001 - 配置不可用时退到 cwd，测试用环境变量覆盖
        return Path("models") / "vision"


def available_providers() -> list[str]:
    from src.engine.hardware import get_ort_providers
    try:
        return get_ort_providers()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Vision] provider 探测失败，退回 CPU: %s", exc)
        return ["CPUExecutionProvider"]


def get_session(spec: ModelSpec, model_dir: Path | None = None):
    """返回缓存的 InferenceSession；文件缺失或加载失败时返回 None。"""
    import onnxruntime as ort

    directory = Path(model_dir) if model_dir is not None else default_model_dir()
    path = directory / spec.filename
    if not path.is_file():
        logger.info("[Vision] 缺少模型文件，跳过 %s: %s", spec.key, path)
        return None
    providers = available_providers()
    cache_key = (str(path), tuple(providers))
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = min(4, os.cpu_count() or 2)
        session = ort.InferenceSession(str(path), options, providers=providers)
    except Exception as exc:  # noqa: BLE001
        logger.error("[Vision] 模型加载失败 %s: %s", spec.key, exc)
        return None
    with _CACHE_LOCK:
        _MODEL_CACHE[cache_key] = session
    logger.info("[Vision] 已加载 %s (%s) providers=%s", spec.key, spec.arch,
                session.get_providers())
    return session


def model_status(model_dir: Path | None = None) -> list[dict]:
    """模型文件清单（供 /health 与排障用）。"""
    directory = Path(model_dir) if model_dir is not None else default_model_dir()
    rows = []
    for spec in SPECS:
        path = directory / spec.filename
        cached = cached_session_status(spec, directory)
        rows.append({
            "key": spec.key,
            "arch": spec.arch,
            "file": spec.filename,
            "present": path.is_file(),
            "size_mb": round(path.stat().st_size / (1024 * 1024), 1) if path.is_file() else 0.0,
            "license": spec.license,
            "license_verified": spec.licensed,
            "license_note": ("标准许可证已登记" if spec.licensed else
                             "来源页标记为 other；本地非商用配置可启用，具体条款待核实"),
            "source": spec.source,
            "classes": sorted(set(spec.mapping.values())),
            **cached,
        })
    return rows


def cached_session_status(spec: ModelSpec, model_dir: Path | None = None) -> dict:
    """Describe a loaded session without loading model weights as a side effect."""
    directory = Path(model_dir) if model_dir is not None else default_model_dir()
    path = directory / spec.filename
    with _CACHE_LOCK:
        sessions = [session for (cached_path, _providers), session in _MODEL_CACHE.items()
                    if cached_path == str(path)]
        last_provider = _LAST_PROVIDER.get(spec.key)
    if not sessions:
        return {"session_loaded": False, "primary_provider": None,
                "provider_chain": [], "last_provider": last_provider}
    try:
        providers = list(sessions[-1].get_providers())
    except Exception:  # noqa: BLE001
        providers = []
    primary = next((provider for provider in providers
                    if provider != "CPUExecutionProvider"),
                   providers[0] if providers else None)
    return {"session_loaded": True, "primary_provider": primary,
            "provider_chain": providers, "last_provider": last_provider}


def record_provider(spec_key: str, providers: list[str]) -> None:
    """Keep the preferred provider from the latest successful inference for diagnostics."""
    primary = next((provider for provider in providers
                    if provider != "CPUExecutionProvider"),
                   providers[0] if providers else "")
    if primary:
        with _CACHE_LOCK:
            _LAST_PROVIDER[str(spec_key)] = primary


def clear_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
