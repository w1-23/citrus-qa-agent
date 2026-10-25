"""Vision (image-based citrus disease screening) subpackage.

Runtime pieces:
- ``context``:   request-scoped image attachments (set by the API entry point)
- ``preprocess``: decode / leaf-crop / letterbox views (PIL + numpy only)
- ``models``:    ONNX model specs, session cache, provider chain
- ``engine``:    per-model inference, cross-model fusion, abstain rules, report text

The package never trains or downloads anything at import time; the ONNX files are
shipped by ``04_脚本_scripts/prepare_models.py --vision``.
"""
from src.vision.engine import VisionEngine, get_engine  # noqa: F401
