"""柑橘病害标签库。

单一事实来源：模型的原始类别名 → 规范 key → 中文名 / 知识库检索词。
工具只认这里的 key，模型文件只通过 ``models.ModelSpec.mapping`` 映射进来。

覆盖口径（重要，工具输出必须如实转述）：
- ``tier == "citrus"``：训练样本来自柑橘（果实/叶片），可直接作为初筛结论；
- ``tier == "weak"``：来自非柑橘样本（如南瓜白粉病）的泛化标签，只作弱参考；
- ``UNCOVERED``：已知但当前三个模型都识别不了的类别，工具必须说明“不在范围内”，
  避免把缺素/虫害静默判成“健康”（冒烟测试中 42 张未覆盖图片里有 34 张被判成健康）。
"""
from __future__ import annotations

from dataclasses import dataclass

HEALTHY = "healthy"


@dataclass(frozen=True)
class DiseaseLabel:
    key: str
    zh: str
    en: str
    kb_terms: tuple[str, ...]
    tier: str = "citrus"
    note: str = ""


LABELS: dict[str, DiseaseLabel] = {
    "canker": DiseaseLabel(
        "canker", "柑橘溃疡病", "Citrus canker",
        ("柑橘溃疡病", "canker", "Xanthomonas citri", "溃疡病防治")),
    "greening": DiseaseLabel(
        "greening", "柑橘黄龙病", "Citrus greening (HLB)",
        ("柑橘黄龙病", "huanglongbing", "HLB", "Candidatus Liberibacter", "木虱")),
    "black_spot": DiseaseLabel(
        "black_spot", "柑橘黑斑病", "Citrus black spot",
        ("柑橘黑斑病", "black spot", "Guignardia citricarpa", "Phyllosticta citricarpa")),
    "melanose": DiseaseLabel(
        "melanose", "柑橘黑点病（梅色斑病）", "Citrus melanose",
        ("柑橘黑点病", "melanose", "Diaporthe citri", "砂皮病")),
    "sooty_mould": DiseaseLabel(
        "sooty_mould", "煤烟病", "Sooty mould",
        ("柑橘煤烟病", "sooty mould", "sooty mold", "Capnodium")),
    "dry_leaf": DiseaseLabel(
        "dry_leaf", "叶片枯焦（生理/环境胁迫）", "Dry leaf",
        ("柑橘叶片枯焦", "dry leaf", "叶尖枯", "日灼")),
    "leaf_curl_virus": DiseaseLabel(
        "leaf_curl_virus", "叶卷曲病毒病", "Leaf curl virus",
        ("柑橘卷叶", "leaf curl", "病毒病", "柑橘病毒")),
    "spider_mites": DiseaseLabel(
        "spider_mites", "红蜘蛛（叶螨）为害", "Spider mites",
        ("柑橘红蜘蛛", "spider mite", "Tetranychus", "叶螨防治")),
    "bacterial_blight": DiseaseLabel(
        "bacterial_blight", "柑橘细菌性疫病", "Bacterial blight",
        ("柑橘细菌性病害", "bacterial blight", "细菌性疫病")),
    "powdery_mildew": DiseaseLabel(
        "powdery_mildew", "白粉病（泛化标签）", "Powdery mildew",
        ("柑橘白粉病", "powdery mildew"),
        tier="weak",
        note="该标签来自 PlantVillage 的非柑橘样本，仅作弱参考"),
    HEALTHY: DiseaseLabel(
        HEALTHY, "未见明显病斑（健康）", "Healthy",
        ("柑橘健康叶片", "healthy citrus leaf"),
        note="只在未见病斑时给出；缺素、虫害、药害可能同样表现为“看起来健康”"),
}

#: 工具可以给出结论的类别（顺序即展示顺序）。
COVERED_KEYS: tuple[str, ...] = tuple(LABELS)

#: 柑橘样本训练的类别（可直接当初筛结论）。
CITRUS_KEYS: tuple[str, ...] = tuple(k for k in LABELS if LABELS[k].tier == "citrus")

#: 非柑橘样本训练的泛化标签（只作弱参考，输出时必须单独标注）。
WEAK_KEYS: tuple[str, ...] = tuple(k for k in LABELS if LABELS[k].tier == "weak")

#: 已知但当前模型覆盖不到的类别：只能提示“不在识别范围内”。
UNCOVERED: tuple[tuple[str, str], ...] = (
    ("scab", "疮痂病"),
    ("greasy_spot", "脂点黄斑病"),
    ("anthracnose", "炭疽病"),
    ("die_back", "枝枯/衰退"),
    ("shot_hole", "细菌性穿孔/角斑"),
    ("spiny_whitefly", "粉虱为害"),
    ("mealybugs", "蚧壳虫"),
    ("citrus_leafminer", "潜叶蛾"),
    ("red_scale", "红圆蚧"),
    ("texas_mite", "得州螨"),
    ("fe_deficiency", "缺铁"),
    ("mg_deficiency", "缺镁"),
    ("zn_deficiency", "缺锌"),
    ("mn_deficiency", "缺锰"),
    ("n_deficiency", "缺氮"),
    ("tristeza", "衰退病"),
)


def label(key: str) -> DiseaseLabel | None:
    return LABELS.get(str(key or "").strip().lower())


def zh_name(key: str) -> str:
    item = label(key)
    return item.zh if item else str(key)


def is_weak(key: str) -> bool:
    item = label(key)
    return bool(item and item.tier == "weak")


def kb_terms(key: str) -> tuple[str, ...]:
    item = label(key)
    return item.kb_terms if item else ()


def covered_zh_list(keys: tuple[str, ...] | None = None) -> str:
    seq = COVERED_KEYS if keys is None else keys
    return "、".join(LABELS[k].zh for k in seq if k in LABELS)


def uncovered_zh_list(limit: int = 16) -> str:
    names = [zh for _key, zh in UNCOVERED[:limit]]
    return "、".join(names)
