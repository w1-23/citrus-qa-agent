"""分阶段模型推理、跨模型融合、拒识规则和可诊断输出。"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.vision import labels as label_bank
from src.vision import models as model_registry
from src.vision.preprocess import (
    decode_image, green_box, image_quality_issues, supplemental_views, to_input,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE = 0.60
DEFAULT_MARGIN = 0.12
DEFAULT_HEALTHY_VETO = 0.25


#: 实际执行提供器按“硬件加速优先”展示，而不是按字母序
_PROVIDER_ORDER = ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")


def _ordered_providers(names) -> list[str]:
    def rank(name: str) -> tuple[int, str]:
        try:
            return (_PROVIDER_ORDER.index(name), name)
        except ValueError:
            return (len(_PROVIDER_ORDER), name)
    return sorted(set(names), key=rank)


def _softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    arr = arr - float(np.max(arr))
    exp = np.exp(arr)
    total = float(exp.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    return exp / total


@dataclass
class Diagnosis:
    status: str
    top: list[tuple[str, float]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    models_used: list[str] = field(default_factory=list)
    models_missing: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    views: int = 0
    inference_runs: int = 0
    model_runs: list[dict] = field(default_factory=list)
    quality_issues: list[str] = field(default_factory=list)
    stage2_triggered: bool = False
    elapsed_ms: int = 0
    image_size: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "top": [{"key": k, "zh": label_bank.zh_name(k), "prob": round(p, 4)}
                    for k, p in self.top],
            "models_used": list(self.models_used),
            "models_missing": list(self.models_missing),
            "providers": list(self.providers),
            "views": self.views,
            "inference_runs": self.inference_runs,
            "model_runs": list(self.model_runs),
            "quality_issues": list(self.quality_issues),
            "stage2_triggered": self.stage2_triggered,
            "elapsed_ms": self.elapsed_ms,
            "image_size": list(self.image_size),
            "notes": list(self.notes),
            "error": self.error,
        }


class VisionEngine:
    def __init__(self, model_dir: Path | str | None = None,
                 confidence_threshold: float = DEFAULT_CONFIDENCE,
                 margin: float = DEFAULT_MARGIN,
                 healthy_veto: float = DEFAULT_HEALTHY_VETO,
                 allow_unlicensed: bool = False):
        self.model_dir = Path(model_dir) if model_dir else None
        self.confidence_threshold = float(confidence_threshold)
        self.margin = float(margin)
        self.healthy_veto = float(healthy_veto)
        self.allow_unlicensed = bool(allow_unlicensed)

    # ── 单模型 ──
    def _view_scores(self, spec, session, crop) -> dict[str, float]:
        input_name = session.get_inputs()[0].name
        tensor = to_input(crop, spec.input_size, spec.mean, spec.std)
        outputs = session.run(None, {input_name: tensor})
        probs = _softmax(np.asarray(outputs[0]).reshape(-1))
        return {key: float(probs[index]) for index, key in spec.mapping.items()
                if index < probs.size}

    def _run_views(self, spec, session, views) -> tuple[dict[str, float], int, int]:
        pooled: dict[str, float] = {}
        runs = 0
        started = time.monotonic()
        for view_name, crop in views:
            try:
                scores = self._view_scores(spec, session, crop)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Vision] %s view=%s 推理失败: %s", spec.key, view_name, exc)
                continue
            runs += 1
            for key, probability in scores.items():
                pooled[key] = max(pooled.get(key, 0.0), probability)
        elapsed = int((time.monotonic() - started) * 1000)
        return pooled, runs, elapsed

    @staticmethod
    def _top(scores: dict[str, float]) -> tuple[str, float, float]:
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        if not ranked:
            return "", 0.0, 0.0
        return ranked[0][0], ranked[0][1], ranked[1][1] if len(ranked) > 1 else 0.0

    def _disagreement_models(self, states: dict[str, dict]) -> set[str]:
        """Return models that disagree on a confident class both models support."""
        rows = list(states.items())
        involved: set[str] = set()
        for index, (left_key, left) in enumerate(rows):
            for right_key, right in rows[index + 1:]:
                shared = (set(left["spec"].mapping.values())
                          & set(right["spec"].mapping.values()))
                if not shared:
                    continue
                left_top = self._top({k: v for k, v in left["scores"].items() if k in shared})
                right_top = self._top({k: v for k, v in right["scores"].items() if k in shared})
                left_label, left_prob, left_second = left_top
                right_label, right_prob, right_second = right_top
                if (left_label and right_label and left_label != right_label
                        and left_prob >= self.confidence_threshold
                        and right_prob >= self.confidence_threshold
                        and left_prob - left_second >= self.margin
                        and right_prob - right_second >= self.margin):
                    involved.update((left_key, right_key))
        return involved

    # ── 融合 ──
    def _fuse(self, per_model: list[tuple[object, dict[str, float]]]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for spec, scores in per_model:
            weight = getattr(spec, "weight", 1.0)
            for key, prob in scores.items():
                contribution = max(0.0, min(1.0, weight * float(prob)))
                fused[key] = 1.0 - (1.0 - fused.get(key, 0.0)) * (1.0 - contribution)
        total = sum(fused.values())
        if total <= 0:
            return {}
        return {key: value / total for key, value in fused.items()}

    # ── 对外 ──
    def diagnose_image(self, img) -> Diagnosis:
        started = time.monotonic()
        quality = image_quality_issues(img)
        if quality:
            return Diagnosis(status="low_quality", image_size=tuple(img.size),
                             quality_issues=quality,
                             elapsed_ms=int((time.monotonic() - started) * 1000))

        providers: set[str] = set()
        specs = model_registry.enabled_specs(self.allow_unlicensed)
        skipped = [spec.key for spec in model_registry.SPECS if spec not in specs]
        states: dict[str, dict] = {}
        missing: list[str] = []
        for spec in specs:
            session = model_registry.get_session(spec, self.model_dir)
            if session is None:
                missing.append(spec.key)
                continue
            session_providers = list(session.get_providers())
            providers.update(session_providers)
            if session_providers:
                model_registry.record_provider(spec.key, session_providers)
            scores, runs, elapsed = self._run_views(spec, session, [("full", img)])
            if not scores:
                missing.append(spec.key)
                continue
            states[spec.key] = {
                "spec": spec, "session": session, "scores": scores,
                "initial_top": self._top(scores)[0], "runs": runs,
                "elapsed_ms": elapsed, "followup_views": [],
                "followup_reasons": [],
                "primary_provider": session_providers[0] if session_providers else None,
            }

        if not states:
            return Diagnosis(
                status="no_model",
                models_missing=missing,
                providers=_ordered_providers(providers),
                elapsed_ms=int((time.monotonic() - started) * 1000),
                image_size=tuple(img.size),
                error="未找到可用的病害识别模型文件",
            )

        disagreement_models = self._disagreement_models(states)
        initial_disagreement = bool(disagreement_models)
        target_unclear = green_box(img) is None
        followup_models: set[str] = set()
        for key, state in states.items():
            _top_key, top_prob, second_prob = self._top(state["scores"])
            if top_prob < self.confidence_threshold:
                followup_models.add(key)
                state["followup_reasons"].append("low_confidence")
            if top_prob - second_prob < self.margin:
                followup_models.add(key)
                state["followup_reasons"].append("small_margin")
            if target_unclear:
                followup_models.add(key)
                state["followup_reasons"].append("target_unclear")
        followup_models.update(disagreement_models)
        for key in disagreement_models:
            states[key]["followup_reasons"].append("shared_label_disagreement")

        followup_views = supplemental_views(img, include_tiles=target_unclear)
        if followup_models and followup_views:
            for key in followup_models:
                state = states[key]
                scores, runs, elapsed = self._run_views(
                    state["spec"], state["session"], followup_views)
                for label, probability in scores.items():
                    state["scores"][label] = max(state["scores"].get(label, 0.0), probability)
                state["runs"] += runs
                state["elapsed_ms"] += elapsed
                state["followup_views"] = [name for name, _crop in followup_views]

        per_model = [(state["spec"], state["scores"]) for state in states.values()]
        used = list(states)
        fused = self._fuse(per_model)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        top = ranked[:3]
        top1_key, top1_prob = ranked[0] if ranked else ("", 0.0)
        top2_prob = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top1_prob - top2_prob
        disease_signal = max([value for key, value in fused.items()
                              if key != label_bank.HEALTHY] or [0.0])

        notes: list[str] = []
        if skipped:
            notes.append(
                "按许可策略未加载：" + ", ".join(skipped) +
                "（模型许可未核实；按当前配置跳过）")
        status = "ok"
        if top1_prob < self.confidence_threshold:
            status = "uncertain"
            notes.append(
                f"最高置信度只有 {top1_prob:.2f}（阈值 {self.confidence_threshold:.2f}），"
                "图像证据不足以支撑单一结论")
        if margin < self.margin:
            status = "uncertain"
            notes.append(
                f"最高分与次高分过于接近（{top1_prob:.2f} vs {top2_prob:.2f}），"
                "三个模型之间没有形成一致判断")
        if top1_key == label_bank.HEALTHY:
            status = "uncertain"
            notes.append(
                "模型首选“未见明显病斑”，但当前模型没有范围外检测能力，"
                "不能据此确认植株健康；缺素、虫害、药害及未覆盖病害也可能呈现类似外观")
            if disease_signal >= self.healthy_veto:
                notes.append(f"同时仍有 {disease_signal:.2f} 的病害信号")
        final_disagreement = bool(self._disagreement_models(states))
        if final_disagreement:
            status = "uncertain"
            notes.append("全图与局部复核后的模型首选类别仍不一致，按不确定结果处理")
        if top1_key and label_bank.is_weak(top1_key):
            notes.append(label_bank.label(top1_key).note)

        return Diagnosis(
            status=status,
            top=top,
            scores=fused,
            models_used=used,
            models_missing=missing,
            providers=_ordered_providers(providers),
            views=1 + (len(followup_views) if followup_models else 0),
            inference_runs=sum(state["runs"] for state in states.values()),
            model_runs=[{
                "model": key,
                "initial_views": ["full"],
                "followup_views": list(state["followup_views"]),
                "followup_reasons": list(state["followup_reasons"]),
                "run_count": state["runs"],
                "elapsed_ms": state["elapsed_ms"],
                "primary_provider": state["primary_provider"],
            } for key, state in states.items()],
            stage2_triggered=bool(followup_models and followup_views),
            elapsed_ms=int((time.monotonic() - started) * 1000),
            image_size=tuple(img.size),
            notes=notes,
        )

    def diagnose_bytes(self, payload: bytes) -> Diagnosis:
        try:
            img = decode_image(payload)
        except Exception as exc:  # noqa: BLE001
            return Diagnosis(status="error", error=str(exc))
        try:
            return self.diagnose_image(img)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[Vision] 推理异常")
            return Diagnosis(status="error", error=f"{type(exc).__name__}: {exc}")

    def diagnose_data_url(self, data_url: str) -> Diagnosis:
        payload = data_url_to_bytes(data_url)
        if payload is None:
            return Diagnosis(status="error", error="图片数据格式无法解析（需要 data: URL）")
        return self.diagnose_bytes(payload)


def data_url_to_bytes(data_url: str) -> bytes | None:
    text = str(data_url or "")
    if not text.startswith("data:") or "," not in text:
        return None
    _header, encoded = text.split(",", 1)
    try:
        return base64.b64decode(encoded, validate=False)
    except Exception:  # noqa: BLE001
        return None


_ENGINE: VisionEngine | None = None


def get_engine() -> VisionEngine:
    """进程级单例（会话缓存本身在 models 层）。"""
    global _ENGINE
    if _ENGINE is None:
        model_dir = None
        confidence = DEFAULT_CONFIDENCE
        margin = DEFAULT_MARGIN
        veto = DEFAULT_HEALTHY_VETO
        allow_unlicensed = False
        try:
            from src.config import settings
            model_dir = str(getattr(settings, "VISION_MODEL_DIR", "") or "") or None
            confidence = float(getattr(settings, "VISION_CONFIDENCE_THRESHOLD", confidence))
            margin = float(getattr(settings, "VISION_MARGIN_THRESHOLD", margin))
            veto = float(getattr(settings, "VISION_HEALTHY_VETO", veto))
            allow_unlicensed = bool(getattr(settings, "VISION_ALLOW_UNLICENSED", False))
        except Exception:  # noqa: BLE001
            pass
        _ENGINE = VisionEngine(model_dir=model_dir, confidence_threshold=confidence,
                               margin=margin, healthy_veto=veto,
                               allow_unlicensed=allow_unlicensed)
    return _ENGINE


def reset_engine() -> None:
    global _ENGINE
    _ENGINE = None
    model_registry.clear_cache()


# ── 报告 ──
def format_report(diag: Diagnosis, *, image_name: str = "图片") -> str:
    """把诊断结果写成模型可读的中文报告（工具返回值）。"""
    head = (f"【图像病害初筛】{image_name} · {diag.image_size[0]}×{diag.image_size[1]} · "
            f"{diag.elapsed_ms} ms · 模型 {len(diag.models_used)} 个"
            f"（{', '.join(diag.models_used) or '无'}）")
    if diag.providers:
        head += f" · EP {diag.providers[0]}"

    if diag.status == "error":
        return f"{head}\n状态：识别失败（{diag.error}）。请提示用户换一张更清晰的叶片近景照片。"

    if diag.status == "no_model":
        return (f"{head}\n状态：本机未安装病害识别模型（{', '.join(diag.models_missing)}），"
                "无法给出图像结论。请说明“图像识别模型未就绪”，不要凭外观下诊断。")

    if diag.status == "low_quality":
        return (f"{head}\n图片质量不足，未运行病害识别：" + "；".join(diag.quality_issues)
                + "。请上传清晰、光线充足、靠近目标部位的照片。")

    lines = [head]
    if diag.status == "ok":
        key, prob = diag.top[0]
        lines.append(f"结论：疑似「{label_bank.zh_name(key)}」（模型评分 {prob:.0%}）")
    else:
        lines.append("结论：无法确定（图像证据不足或模型间不一致），不给单一病名")

    if diag.top:
        ranking = "；".join(f"{label_bank.zh_name(k)} {p:.0%}" for k, p in diag.top)
        lines.append(f"候选排序（模型评分）：{ranking}")

    lines.append(f"可识别范围（{len(label_bank.CITRUS_KEYS)} 类，柑橘样本训练）："
                 f"{label_bank.covered_zh_list(label_bank.CITRUS_KEYS)}")
    if label_bank.WEAK_KEYS:
        lines.append("弱参考标签（非柑橘样本训练，仅供参考，不可当作初筛结论）："
                     f"{label_bank.covered_zh_list(label_bank.WEAK_KEYS)}")
    lines.append(f"不在范围内（需人工判断）：{label_bank.uncovered_zh_list()}")

    for note in diag.notes:
        lines.append(f"注意：{note}")

    if diag.model_runs:
        reason_names = {
            "low_confidence": "全图分数低",
            "small_margin": "全图候选接近",
            "target_unclear": "未定位叶片区域",
            "shared_label_disagreement": "共同类别意见不一致",
        }
        detail = "；".join(
            f"{row['model']} {row['run_count']} 次/{row['elapsed_ms']} ms"
            + (f"（补充视图：{','.join(row['followup_views'])}）"
               if row["followup_views"] else "（仅全图）")
            + (f"；复核原因：{'、'.join(reason_names.get(x, x) for x in row['followup_reasons'])}"
               if row["followup_reasons"] else "")
            for row in diag.model_runs
        )
        lines.append(f"推理轨迹：{detail}；共 {diag.inference_runs} 次模型推理")

    if diag.status == "ok":
        key = diag.top[0][0]
        if label_bank.is_weak(key):
            lines.append("提示：命中的是弱参考标签，只能作为“有没有这类病斑迹象”的弱提示，"
                         "必须请用户补充症状描述或做人工复核。")
        if key != label_bank.HEALTHY:
            terms = "、".join(label_bank.kb_terms(key)[:3])
            lines.append(f"知识库检索建议：{terms}")
        else:
            lines.append("知识库检索建议：健康叶片管理、柑橘缺素症鉴别")

    lines.append("使用要求：这是图像初筛，不是诊断结论；必须说明“需结合症状描述、"
                 "栽培环境与专家复核”，不得写成确诊，也不得据此推荐农药用量。")
    return "\n".join(lines)
