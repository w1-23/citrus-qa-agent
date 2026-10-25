"""图像预处理：解码 → 叶部前景框 → letterbox 多视图（不居中裁剪）。

设计约定（与冒烟测试结论一致）：
- **不裁剪主体**：病虫害在图上的位置不固定（叶尖、叶缘、叶背），居中裁剪会把
  病斑裁掉。统一 letterbox 缩放后整幅送入，另加若干局部视图做“投票”。
- **多视图**：整幅 + 叶部前景框 + 中央 72% + 四象限 60%，最多 7 个视图；
  单视图漏检的部位由其它视图补回，概率按视图取最大后归一。
- 只依赖 PIL + numpy，不引入 torch / torchvision。
"""
from __future__ import annotations

import io
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

#: 解码后长边上限（4032×3024 的实拍图降到 1600 已足够，且更快）。
MAX_SIDE = 1600

#: 单张图片最大字节数（与 API 附件限制一致）。
MAX_BYTES = 20 * 1024 * 1024
PROBE_SIZE = 96
LEAF_PADDING = 0.08
MIN_IMAGE_SIDE = 192
MIN_BRIGHTNESS = 22.0
MIN_LAPLACIAN_VARIANCE = 8.0


class ImageError(ValueError):
    """无法作为图像处理的输入。"""


def decode_image(payload: bytes) -> Image.Image:
    """把原始字节解码成 RGB 图（自动转正 EXIF 方向并限制尺寸）。"""
    if not payload:
        raise ImageError("空图片数据")
    if len(payload) > MAX_BYTES:
        raise ImageError(f"图片超过 {MAX_BYTES // (1024 * 1024)}MB 上限")
    try:
        img = Image.open(io.BytesIO(payload))
        img.load()
    except Exception as exc:  # noqa: BLE001 - 任何解码失败都归为无法识别
        raise ImageError(f"图片解码失败: {type(exc).__name__}") from exc
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        scale = MAX_SIDE / float(max(img.size))
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                         Image.BILINEAR)
    if img.width < 32 or img.height < 32:
        raise ImageError("图片尺寸过小（<32px）")
    return img


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest 8-connected component from the small leaf mask."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    largest: list[tuple[int, int]] = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        component: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(component) > len(largest):
            largest = component
    result = np.zeros(mask.shape, dtype=bool)
    if largest:
        ys, xs = zip(*largest)
        result[np.asarray(ys), np.asarray(xs)] = True
    return result


def green_box(img: Image.Image) -> Tuple[int, int, int, int] | None:
    """用 HSV 色域、闭运算和最大连通域定位叶片候选框。"""
    probe = img.copy()
    probe.thumbnail((PROBE_SIZE, PROBE_SIZE), Image.BILINEAR)
    hsv = np.asarray(probe.convert("HSV"), dtype=np.uint8)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # PIL hue is [0, 255]; include yellow-green chlorosis as well as ordinary green.
    mask = (hue >= 18) & (hue <= 112) & (saturation >= 35) & (value >= 28)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask = np.asarray(mask_image) > 0
    ratio = float(mask.mean())
    if ratio < 0.03 or ratio > 0.97:
        return None
    mask = _largest_component(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) < 24:
        return None
    sx = img.width / probe.width
    sy = img.height / probe.height
    pad_x = LEAF_PADDING * (xs.max() - xs.min() + 1) * sx
    pad_y = LEAF_PADDING * (ys.max() - ys.min() + 1) * sy
    x0 = max(0, int(xs.min() * sx - pad_x))
    y0 = max(0, int(ys.min() * sy - pad_y))
    x1 = min(img.width, int((xs.max() + 1) * sx + pad_x))
    y1 = min(img.height, int((ys.max() + 1) * sy + pad_y))
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    if (x1 - x0) * (y1 - y0) < 0.04 * img.width * img.height:
        return None
    return x0, y0, x1, y1


def _crop(img: Image.Image, box: Iterable[float]) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def image_quality_issues(img: Image.Image) -> list[str]:
    """Return capture-quality warnings; disease inference must not override them."""
    issues: list[str] = []
    if min(img.size) < MIN_IMAGE_SIDE:
        issues.append(f"短边只有 {min(img.size)} px，目标细节可能不足")
    sample = img.copy()
    sample.thumbnail((256, 256), Image.BILINEAR)
    gray = np.asarray(sample.convert("L"), dtype=np.float32)
    if float(gray.mean()) < MIN_BRIGHTNESS:
        issues.append("图片整体过暗")
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        center = gray[1:-1, 1:-1]
        laplacian = (-4.0 * center + gray[:-2, 1:-1] + gray[2:, 1:-1]
                     + gray[1:-1, :-2] + gray[1:-1, 2:])
        if float(laplacian.var()) < MIN_LAPLACIAN_VARIANCE:
            issues.append("图片细节较少或对焦模糊")
    return issues


def supplemental_views(img: Image.Image, *, include_tiles: bool) -> List[Tuple[str, Image.Image]]:
    """Construct second-stage views only after the full-image pass requests them."""
    views: List[Tuple[str, Image.Image]] = []
    box = green_box(img)
    if box is not None:
        views.append(("leaf", img.crop(box)))
    if include_tiles or box is None:
        views.extend([
            ("tl", _crop(img, (0.0, 0.0, 0.6, 0.6))),
            ("tr", _crop(img, (0.4, 0.0, 1.0, 0.6))),
            ("bl", _crop(img, (0.0, 0.4, 0.6, 1.0))),
            ("br", _crop(img, (0.4, 0.4, 1.0, 1.0))),
        ])
    return views


def view_crops(img: Image.Image) -> List[Tuple[str, Image.Image]]:
    """Compatibility helper returning the full image and all supplemental views."""
    return [("full", img), *supplemental_views(img, include_tiles=True)]


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """等比缩放到 size×size，以缩放图边缘中位数补边（不裁剪）。"""
    w, h = img.size
    scale = size / float(max(w, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.BICUBIC)
    pixels = np.asarray(resized, dtype=np.uint8)
    border = np.concatenate((pixels[0, :, :], pixels[-1, :, :],
                             pixels[:, 0, :], pixels[:, -1, :]), axis=0)
    background = tuple(int(v) for v in np.median(border, axis=0))
    canvas = Image.new("RGB", (size, size), background)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def to_input(img: Image.Image, size: int, mean, std) -> np.ndarray:
    """生成 (1, 3, H, W) 的 float32 张量（ImageNet 归一化参数由模型元数据给出）。"""
    canvas = letterbox(img, size)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean_arr) / std_arr
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...])
