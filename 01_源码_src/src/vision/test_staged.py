from __future__ import annotations

import contextvars
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.vision import models
from src.vision.engine import VisionEngine
from src.vision.context import get_request_images, reset_request_images, set_request_images
from src.vision.preprocess import image_quality_issues, letterbox, supplemental_views


class FakeSession:
    def __init__(self, spec, preferred_label: str, *, uncertain: bool = False):
        self.spec = spec
        self.preferred_label = preferred_label
        self.uncertain = uncertain
        self.calls = 0

    def get_inputs(self):
        return [type("Input", (), {"name": "pixels"})()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _outputs, _inputs):
        self.calls += 1
        logits = np.zeros(max(len(self.spec.labels), max(self.spec.mapping) + 1), dtype=np.float32)
        if not self.uncertain:
            index = next(i for i, label in self.spec.mapping.items()
                         if label == self.preferred_label)
            logits[:] = -8
            logits[index] = 8
        return [logits[None, :]]


def _leaf_photo(size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), (205, 205, 205))
    draw = ImageDraw.Draw(image)
    draw.ellipse((85, 110, 420, 400), fill=(45, 125, 48), outline=(15, 70, 25), width=5)
    draw.line((120, 280, 385, 215), fill=(220, 220, 180), width=7)
    return image


def _engine(monkeypatch, specs, labels_by_key, *, uncertain=()):
    sessions = {spec.key: FakeSession(spec, labels_by_key[spec.key],
                                     uncertain=spec.key in uncertain)
                for spec in specs}
    monkeypatch.setattr(models, "enabled_specs", lambda _allow: tuple(specs))
    monkeypatch.setattr(models, "get_session", lambda spec, _directory: sessions[spec.key])
    engine = VisionEngine(model_dir=Path("unused"), allow_unlicensed=True)
    return engine, sessions


def test_confident_single_model_runs_only_full_image(monkeypatch):
    spec = models.SPECS[0]
    engine, sessions = _engine(monkeypatch, [spec], {spec.key: "canker"})

    result = engine.diagnose_image(_leaf_photo())

    assert result.status == "ok"
    assert result.stage2_triggered is False
    assert result.inference_runs == 1
    assert sessions[spec.key].calls == 1
    assert result.model_runs[0]["initial_views"] == ["full"]
    assert result.model_runs[0]["followup_views"] == []


def test_low_confidence_adds_leaf_view_only_for_uncertain_model(monkeypatch):
    spec = models.SPECS[0]
    engine, sessions = _engine(monkeypatch, [spec], {spec.key: "canker"},
                               uncertain=[spec.key])

    result = engine.diagnose_image(_leaf_photo())

    assert result.stage2_triggered is True
    assert result.model_runs[0]["followup_views"] == ["leaf"]
    assert sessions[spec.key].calls == 2


def test_healthy_prediction_is_never_reported_as_confirmed_health(monkeypatch):
    spec = models.SPECS[0]
    engine, _sessions = _engine(monkeypatch, [spec], {spec.key: "healthy"})

    result = engine.diagnose_image(_leaf_photo())

    assert result.status == "uncertain"
    assert any("不能据此确认植株健康" in note for note in result.notes)


def test_model_disagreement_reruns_only_models_with_shared_conflicting_labels(monkeypatch):
    convnext, dima, plantvillage = models.SPECS
    specs = [convnext, dima, plantvillage]
    engine, sessions = _engine(monkeypatch, specs, {
        "convnext": "canker", "dima": "healthy", "plantvillage": "greening",
    })

    result = engine.diagnose_image(_leaf_photo())

    assert result.status == "uncertain"
    assert result.stage2_triggered is True
    assert sessions["convnext"].calls > 1
    assert sessions["dima"].calls > 1
    assert sessions["plantvillage"].calls == 1


def test_low_quality_image_skips_all_models(monkeypatch):
    spec = models.SPECS[0]
    engine, sessions = _engine(monkeypatch, [spec], {spec.key: "canker"})

    result = engine.diagnose_image(Image.new("RGB", (512, 512), (0, 0, 0)))

    assert result.status == "low_quality"
    assert result.quality_issues
    assert sessions[spec.key].calls == 0


def test_preprocessing_has_bounded_views_and_edge_median_letterbox():
    image = _leaf_photo()
    assert not image_quality_issues(image)
    assert [name for name, _ in supplemental_views(image, include_tiles=False)] == ["leaf"]
    assert [name for name, _ in supplemental_views(image, include_tiles=True)] == [
        "leaf", "tl", "tr", "bl", "br",
    ]
    portrait = Image.new("RGB", (100, 200), (90, 120, 150))
    boxed = letterbox(portrait, 224)
    assert boxed.size == (224, 224)
    assert boxed.getpixel((0, 0)) == (90, 120, 150)


def test_image_context_isolated_and_reset_per_request():
    first_context = contextvars.copy_context()
    second_context = contextvars.copy_context()
    first_token = first_context.run(set_request_images, [{"name": "first", "url": "data:image/png;base64,YQ=="}])
    second_token = second_context.run(set_request_images, [{"name": "second", "url": "data:image/png;base64,Yg=="}])

    assert first_context.run(lambda: get_request_images()[0].name) == "first"
    assert second_context.run(lambda: get_request_images()[0].name) == "second"
    first_context.run(reset_request_images, first_token)
    second_context.run(reset_request_images, second_token)
    assert first_context.run(get_request_images) == ()
    assert second_context.run(get_request_images) == ()


def test_primary_model_image_capability_comes_from_selected_model(monkeypatch):
    from src.graph import general_agent_graph as graph

    monkeypatch.setattr(graph.settings, "_runtime_model_cfg",
                        {"provider_id": "aliyun_bailian",
                         "main_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"})
    monkeypatch.setattr(graph, "get_deepseek_model", lambda: "deepseek-flash")
    assert graph._main_model_supports_image_input() is True

    monkeypatch.setattr(graph, "get_deepseek_model", lambda: "text-only-model")
    assert graph._main_model_supports_image_input() is False

    monkeypatch.setattr(graph.settings, "_runtime_model_cfg",
                        {"provider_id": "aliyun_bailian",
                         "main_model": "deepseek-v4.1-flash-1017",
                         "main_model_image_input": "false"})
    monkeypatch.setattr(graph, "get_deepseek_model", lambda: "deepseek-v4.1-flash-1017")
    assert graph._main_model_supports_image_input() is True
    assert graph.settings.RESOLVED_MAIN_MODEL_IMAGE_INPUT is True

    monkeypatch.setattr(graph.settings, "_runtime_model_cfg",
                        {"provider_id": "aliyun_bailian",
                         "main_model": "custom-text-model",
                         "main_model_image_input": "true"})
    monkeypatch.setattr(graph, "get_deepseek_model", lambda: "custom-text-model")
    assert graph._main_model_supports_image_input() is False
    assert graph.settings.RESOLVED_MAIN_MODEL_IMAGE_INPUT is False

    monkeypatch.setattr(graph, "get_deepseek_model", lambda: "qwen3-vl-plus")
    assert graph._main_model_supports_image_input() is True
