import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _odd_kernel(size):
    size = max(3, int(size))
    return size if size % 2 else size + 1


def _to_float_alpha(mask, blur_size):
    alpha = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (_odd_kernel(blur_size), _odd_kernel(blur_size)), 0)
    return np.clip(alpha[..., None], 0.0, 1.0)


def _seam_alpha(mask, band, feather_scale=1.4):
    feather = max(3, int(band * feather_scale))
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (_odd_kernel(feather), _odd_kernel(feather)), 0)

    core_kernel = np.ones((_odd_kernel(max(3, band // 2)), _odd_kernel(max(3, band // 2))), dtype=np.uint8)
    core = cv2.erode(mask, core_kernel, iterations=1)
    alpha[core > 0] = 1.0

    return np.clip(alpha, 0.0, 1.0)


def _structure_protection(image, band):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 45, 130)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    detail = (cv2.magnitude(grad_x, grad_y) > 12.0).astype(np.uint8) * 255

    saturated = cv2.inRange(hsv[:, :, 1], 34, 255)
    ink = cv2.inRange(gray, 0, 235)

    structure = cv2.bitwise_or(edges, cv2.bitwise_and(cv2.bitwise_and(saturated, ink), detail))
    structure = cv2.dilate(structure, np.ones((3, 3), dtype=np.uint8), iterations=1)
    structure = cv2.GaussianBlur(structure.astype(np.float32) / 255.0, (_odd_kernel(max(3, band)), _odd_kernel(max(3, band))), 0)
    return np.clip(structure, 0.0, 1.0)


def _foreground_mask(image, band):
    # Downsample and quantize to find dominant background color
    small = cv2.resize(image, (50, 50), interpolation=cv2.INTER_AREA)
    rounded = (small // 16) * 16
    pixels = rounded.reshape(-1, 3)
    colors, counts = np.unique(pixels, axis=0, return_counts=True)
    
    # If the dominant color covers less than 15% of the image (375 pixels),
    # it's likely a dense full-bleed pattern with no clear solid background.
    if np.max(counts) < 375:
        # Fall back to original morphology-based approach
        structure = _structure_protection(image, max(3, band))
        mask = (structure > 0.14).astype(np.uint8) * 255
    else:
        dominant_bin = colors[np.argmax(counts)]
        mask_bin = np.all((image // 16) * 16 == dominant_bin, axis=2)
        if np.any(mask_bin):
            bg_color = np.median(image[mask_bin], axis=0)
        else:
            bg_color = np.array([128.0, 128.0, 128.0])

        # Distance thresholding
        diff = np.abs(image.astype(np.float32) - bg_color)
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        mask = (dist > 30).astype(np.uint8) * 255

    # Morphology cleanup
    kernel_size = _odd_kernel(max(3, band // 2))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    return mask


def _seam_fragment_mask(seam_mask, foreground, band):
    near_kernel = np.ones((_odd_kernel(max(3, band * 2)), _odd_kernel(max(3, band * 2))), dtype=np.uint8)
    grow_kernel = np.ones((_odd_kernel(max(3, band)), _odd_kernel(max(3, band))), dtype=np.uint8)

    seam_neighborhood = cv2.dilate(seam_mask, near_kernel, iterations=1)
    fragments = cv2.bitwise_and(foreground, seam_neighborhood)
    fragments = cv2.dilate(fragments, grow_kernel, iterations=1)
    fragments = cv2.morphologyEx(fragments, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    return cv2.bitwise_and(fragments, seam_neighborhood)


def _texture_profile(image, band):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)

    structure = _structure_protection(image, band)
    structure_density = float(np.mean(structure > 0.18))
    smooth_ratio = float(np.mean(gradient < 13.0))
    saturation_mean = float(np.mean(hsv[:, :, 1]))
    contrast = float(np.std(gray))

    is_structured = (
        structure_density > 0.04 and smooth_ratio > 0.38
    ) or (
        structure_density > 0.08 and smooth_ratio > 0.25 and saturation_mean > 35.0 and contrast > 28.0
    )

    return {
        "is_structured": bool(is_structured),
        "structure_density": round(structure_density, 4),
        "smooth_ratio": round(smooth_ratio, 4),
        "saturation_mean": round(saturation_mean, 3),
        "contrast": round(contrast, 3),
    }


def _masked_blend(base, repaired, mask, band, strength=1.0):
    if np.count_nonzero(mask) == 0:
        return base

    alpha = _seam_alpha(mask, band, feather_scale=1.2)[..., None] * float(strength)
    result = base.astype(np.float32) * (1.0 - alpha)
    result += repaired.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def _cv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _pil_to_cv(image):
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _resize_for_ai(image, mask):
    h, w = image.shape[:2]
    max_side = int(os.getenv("VERTEX_AI_MAX_SIDE", "1536"))
    side = max(h, w)
    if side <= max_side:
        return image, mask, 1.0

    scale = max_side / side
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask, scale


def _vertex_project_id():
    return (
        os.getenv("VERTEX_AI_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
        or "gen-lang-client-0628424192"
    )


def _vertex_location():
    return os.getenv("VERTEX_AI_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "us-central1"


def _generated_image_to_pil(generated_image, temp_dir):
    image_obj = getattr(generated_image, "image", generated_image)
    image_bytes = (
        getattr(image_obj, "image_bytes", None)
        or getattr(image_obj, "_image_bytes", None)
        or getattr(generated_image, "image_bytes", None)
        or getattr(generated_image, "_image_bytes", None)
    )
    if image_bytes:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    output_path = Path(temp_dir) / "ai_edited.png"
    image_obj.save(location=str(output_path))
    return Image.open(output_path).convert("RGB")


def _vertex_edit_offset(offset, edit_mask):
    if genai is None or genai_types is None:
        return None, {
            "ai_provider": "vertex_ai_imagen",
            "vertex_status": "missing_dependency",
        }

    project_id = _vertex_project_id()
    location = _vertex_location()
    model = os.getenv("VERTEX_AI_IMAGE_EDIT_MODEL", "imagen-3.0-capability-001")
    ai_image, ai_mask, scale = _resize_for_ai(offset, edit_mask)
    mask_rgb = cv2.cvtColor(ai_mask, cv2.COLOR_GRAY2BGR)

    prompt = (
        "Repair only the masked center seam lines in this offset-transformed textile repeat tile. "
        "Continue the existing flowers, leaves, stems, fine linework, texture, and background naturally across the seam. "
        "Do not create border strips, frames, mirrored bands, copied columns, text, logos, or watermarks. "
        "Preserve all unmasked areas exactly and return one edited seamless repeat tile."
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "base.png"
            mask_path = Path(temp_dir) / "mask.png"
            cv2.imwrite(str(base_path), ai_image)
            cv2.imwrite(str(mask_path), mask_rgb)

            client = genai.Client(vertexai=True, project=project_id, location=location)
            raw_ref = genai_types.RawReferenceImage(
                referenceImage=genai_types.Image.from_file(location=str(base_path)),
                referenceId=0,
            )
            mask_ref = genai_types.MaskReferenceImage(
                referenceImage=genai_types.Image.from_file(location=str(mask_path)),
                referenceId=1,
                config=genai_types.MaskReferenceConfig(
                    maskMode="MASK_MODE_USER_PROVIDED",
                    maskDilation=float(os.getenv("VERTEX_AI_MASK_DILATION", "0.02")),
                ),
            )
            response = client.models.edit_image(
                model=model,
                prompt=prompt,
                reference_images=[raw_ref, mask_ref],
                config=genai_types.EditImageConfig(
                    editMode="EDIT_MODE_INPAINT_INSERTION",
                    numberOfImages=int(os.getenv("VERTEX_AI_CANDIDATE_COUNT", "4")),
                    guidanceScale=float(os.getenv("VERTEX_AI_GUIDANCE_SCALE", "21")),
                    outputMimeType="image/png",
                    baseSteps=int(os.getenv("VERTEX_AI_BASE_STEPS", "50")),
                ),
            )
            generated_images = getattr(response, "generated_images", None) or []
            if not generated_images:
                generated_images = getattr(response, "generatedImages", None) or []
            edited_images = [
                _generated_image_to_pil(generated_image, temp_dir)
                for generated_image in generated_images
            ]
    except Exception as exc:
        return None, {
            "ai_provider": "vertex_ai_imagen",
            "vertex_status": "failed",
            "vertex_error": str(exc)[:240],
            "vertex_model": model,
            "vertex_project": project_id,
            "vertex_location": location,
        }

    if not edited_images:
        return None, {
            "ai_provider": "vertex_ai_imagen",
            "vertex_status": "no_image_returned",
            "vertex_model": model,
            "vertex_project": project_id,
            "vertex_location": location,
        }

    edited_cvs = []
    for edited in edited_images:
        edited_cv = _pil_to_cv(edited)
        if edited_cv.shape[:2] != ai_image.shape[:2]:
            edited_cv = cv2.resize(
                edited_cv,
                (ai_image.shape[1], ai_image.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )

        if scale != 1.0:
            edited_cv = cv2.resize(
                edited_cv,
                (offset.shape[1], offset.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        edited_cvs.append(edited_cv)

    return edited_cvs, {
        "ai_provider": "vertex_ai_imagen",
        "vertex_status": "used",
        "vertex_model": model,
        "vertex_project": project_id,
        "vertex_location": location,
        "vertex_scale": round(float(scale), 4),
        "vertex_candidate_count": len(edited_cvs),
    }


def offsetTransform(image):
    h, w = image.shape[:2]
    offset = np.roll(image, w // 2, axis=1)
    offset = np.roll(offset, h // 2, axis=0)
    return offset


def seamMaskDetection(image):
    h, w = image.shape[:2]
    min_dim = min(h, w)
    base_band = max(12, min_dim // 56)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    center_x = w // 2
    center_y = h // 2
    x0 = max(0, center_x - base_band)
    x1 = min(w, center_x + base_band)
    y0 = max(0, center_y - base_band)
    y1 = min(h, center_y + base_band)

    seam_strength = 0.5 * (
        float(np.mean(np.abs(grad_x[:, x0:x1]))) +
        float(np.mean(np.abs(grad_y[y0:y1, :])))
    )
    image_contrast = float(np.std(gray)) + 1e-6
    contrast_factor = np.clip(seam_strength / image_contrast, 0.9, 2.1)
    band = int(np.clip(base_band * contrast_factor, base_band, max(base_band, min_dim // 30)))

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, max(0, center_x - band):min(w, center_x + band)] = 255
    mask[max(0, center_y - band):min(h, center_y + band), :] = 255

    return mask, band


def opencvInpainting(image, mask, radius=None):
    seam_width = max(3, int(np.sqrt(np.count_nonzero(mask) / max(1, sum(image.shape[:2])))))
    if radius is None:
        radius = int(np.clip(seam_width // 2, 3, 8))
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def textureQuiltSeam(image, mask, band, seed=123):
    h, w = image.shape[:2]
    patch = _odd_kernel(np.clip(band * 5, 35, 101))
    half = patch // 2
    step = max(8, band // 2)

    valid = cv2.erode(255 - mask, np.ones((patch, patch), dtype=np.uint8), iterations=1)
    ys, xs = np.where(valid[half:h - half, half:w - half] > 0)
    if len(xs) == 0:
        return image

    candidates = np.column_stack((ys + half, xs + half))
    rng = np.random.default_rng(seed)
    result = image.copy()

    center_x = w // 2
    center_y = h // 2
    targets = []
    for y in range(half, h - half, step):
        targets.append((y, center_x))
    for x in range(half, w - half, step):
        targets.append((center_y, x))

    target_seen = set()
    ordered_targets = []
    for y, x in targets:
        key = (int(y), int(x))
        if key not in target_seen:
            target_seen.add(key)
            ordered_targets.append(key)

    for y, x in ordered_targets:
        y0, y1 = y - half, y + half + 1
        x0, x1 = x - half, x + half + 1
        target_mask = mask[y0:y1, x0:x1] > 0
        if not np.any(target_mask):
            continue

        target_patch = result[y0:y1, x0:x1].astype(np.float32)
        known = ~target_mask
        if np.count_nonzero(known) < patch:
            known = cv2.dilate((known.astype(np.uint8) * 255), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0

        sample_count = min(180, len(candidates))
        sample_ids = rng.choice(len(candidates), size=sample_count, replace=False)
        best_score = None
        best_patch = None

        for idx in sample_ids:
            sy, sx = candidates[idx]
            sy0, sy1 = sy - half, sy + half + 1
            sx0, sx1 = sx - half, sx + half + 1
            source_patch = image[sy0:sy1, sx0:sx1].astype(np.float32)
            diff = source_patch[known] - target_patch[known]
            score = float(np.mean(diff * diff))
            if best_score is None or score < best_score:
                best_score = score
                best_patch = source_patch

        if best_patch is None:
            continue

        paste_alpha = cv2.GaussianBlur(target_mask.astype(np.float32), (_odd_kernel(max(3, band)), _odd_kernel(max(3, band))), 0)
        paste_alpha = np.clip(paste_alpha[..., None], 0.0, 1.0) * 0.92
        target = result[y0:y1, x0:x1].astype(np.float32)
        target = target * (1.0 - paste_alpha) + best_patch * paste_alpha
        result[y0:y1, x0:x1] = np.clip(target, 0, 255).astype(np.uint8)

    return result


def patchTransferSeam(image, mask, band):
    h, w = image.shape[:2]
    patch = _odd_kernel(np.clip(band * 6, 41, 121))
    half = patch // 2
    step = max(6, band // 2)
    search_stride = max(3, band // 3)

    protected = cv2.dilate(mask, np.ones((_odd_kernel(band * 2), _odd_kernel(band * 2)), dtype=np.uint8), iterations=1)
    valid = cv2.erode(255 - protected, np.ones((patch, patch), dtype=np.uint8), iterations=1)
    candidate_y, candidate_x = np.where(valid[half:h - half:search_stride, half:w - half:search_stride] > 0)
    if len(candidate_x) == 0:
        return image

    candidates = np.column_stack((
        candidate_y * search_stride + half,
        candidate_x * search_stride + half,
    ))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 130).astype(np.float32) / 255.0
    result = image.copy()

    center_x = w // 2
    center_y = h // 2
    targets = []
    for y in range(half, h - half, step):
        targets.append((y, center_x))
    for x in range(half, w - half, step):
        targets.append((center_y, x))

    seen = set()
    for y, x in targets:
        key = (int(y), int(x))
        if key in seen:
            continue
        seen.add(key)

        y0, y1 = y - half, y + half + 1
        x0, x1 = x - half, x + half + 1
        target_mask = mask[y0:y1, x0:x1] > 0
        if not np.any(target_mask):
            continue

        known = ~target_mask
        context = cv2.dilate((target_mask.astype(np.uint8) * 255), np.ones((_odd_kernel(max(3, band)), _odd_kernel(max(3, band))), dtype=np.uint8), iterations=1) > 0
        known = known & context
        if np.count_nonzero(known) < patch:
            known = ~target_mask

        target_patch = result[y0:y1, x0:x1].astype(np.float32)
        target_edge = edges[y0:y1, x0:x1]
        weights = (1.0 + target_edge * 3.0)[known][..., None]

        best_score = None
        best_patch = None
        for sy, sx in candidates:
            sy0, sy1 = sy - half, sy + half + 1
            sx0, sx1 = sx - half, sx + half + 1
            source_patch = image[sy0:sy1, sx0:sx1].astype(np.float32)
            diff = (source_patch[known] - target_patch[known]) * weights
            score = float(np.mean(diff * diff))

            # Prefer patches with a similar amount of painted structure in the fill area.
            source_edge = edges[sy0:sy1, sx0:sx1]
            score += float(abs(np.mean(source_edge[target_mask]) - np.mean(target_edge[target_mask])) * 2200.0)
            if best_score is None or score < best_score:
                best_score = score
                best_patch = source_patch

        if best_patch is None:
            continue

        core = cv2.erode(target_mask.astype(np.uint8) * 255, np.ones((3, 3), dtype=np.uint8), iterations=1)
        feather = cv2.GaussianBlur(target_mask.astype(np.float32), (_odd_kernel(max(3, band // 2)), _odd_kernel(max(3, band // 2))), 0)
        feather[core > 0] = 1.0
        paste_alpha = np.clip(feather[..., None], 0.0, 1.0)

        target = result[y0:y1, x0:x1].astype(np.float32)
        target = target * (1.0 - paste_alpha) + best_patch * paste_alpha
        result[y0:y1, x0:x1] = np.clip(target, 0, 255).astype(np.uint8)

    return result


def edgeCompletion(image, mask, band):
    smooth = cv2.bilateralFilter(
        image,
        d=0,
        sigmaColor=10,
        sigmaSpace=max(3, band // 2),
    )

    alpha = _seam_alpha(mask, band, feather_scale=1.0)[..., None]
    structure_guard = 1.0 - _structure_protection(image, band)[..., None] * 0.8
    blend_alpha = alpha * structure_guard * 0.18

    completed = image.astype(np.float32) * (1.0 - blend_alpha)
    completed += smooth.astype(np.float32) * blend_alpha
    return np.clip(completed, 0, 255).astype(np.uint8)


def _laplacian_pyramid(image, levels):
    gaussian = [image.astype(np.float32)]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))

    laplacian = []
    for i in range(levels):
        expanded = cv2.pyrUp(gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0]))
        laplacian.append(gaussian[i] - expanded)
    laplacian.append(gaussian[-1])
    return laplacian


def pyramidBlendVal(foreground, background, mask, band, levels=5):
    blur_size = _odd_kernel(max(5, band * 2))
    alpha = _seam_alpha(mask, band, feather_scale=0.75)

    structure = _structure_protection(foreground, band)
    low_alpha = np.clip(alpha * (1.0 - structure * 0.65), 0.0, 1.0)

    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)

    fg_base = cv2.GaussianBlur(fg, (blur_size, blur_size), 0)
    bg_base = cv2.GaussianBlur(bg, (blur_size, blur_size), 0)
    fg_detail = fg - fg_base

    core = low_alpha > 0.65
    transition = (low_alpha > 0.05) & ~core

    fg_delta = np.mean(np.abs(fg_base - fg), axis=2)
    bg_delta = np.mean(np.abs(bg_base - fg), axis=2)
    choose_fg = (fg_delta <= bg_delta) | core

    chosen_base = np.where(choose_fg[..., None], fg_base, bg_base)
    blended_base = bg_base.copy()
    blended_base[core] = chosen_base[core]

    if np.any(transition):
        transition_alpha = np.clip((low_alpha - 0.05) / 0.6, 0.0, 1.0)[..., None] * 0.18
        blended_base = blended_base * (1.0 - transition_alpha) + chosen_base * transition_alpha

    result = blended_base + fg_detail

    return np.clip(result, 0, 255).astype(np.uint8)


def lumiStats(image, reference, mask, band):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    kernel_size = _odd_kernel(band * 5)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    ring = cv2.dilate(mask, kernel, iterations=1)
    ring = cv2.subtract(ring, mask)

    seam_pixels = mask > 0
    ring_pixels = ring > 0
    if not np.any(seam_pixels) or not np.any(ring_pixels):
        return image

    seam_l = lab[:, :, 0][seam_pixels]
    ring_l = ref_lab[:, :, 0][ring_pixels]

    seam_mean = float(np.mean(seam_l))
    seam_std = float(np.std(seam_l)) + 1e-6
    ring_mean = float(np.mean(ring_l))
    ring_std = float(np.std(ring_l)) + 1e-6

    corrected_l = (lab[:, :, 0] - seam_mean) * (ring_std / seam_std) + ring_mean
    structure = _structure_protection(image, band)
    alpha = _seam_alpha(mask, band, feather_scale=0.8) * (1.0 - structure * 0.75) * 0.18
    lab[:, :, 0] = lab[:, :, 0] * (1.0 - alpha) + corrected_l * alpha

    lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def unsharpMask(image, mask=None, amount=0.35, radius=1.2):
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharp = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    if mask is None:
        return sharp

    h, w = image.shape[:2]
    band_hint = max(8, min(h, w) // 50)
    alpha = _to_float_alpha(mask, band_hint * 5) * 0.85
    result = image.astype(np.float32) * (1.0 - alpha)
    result += sharp.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def reverseOffset(image):
    h, w = image.shape[:2]
    reversed_image = np.roll(image, -(w // 2), axis=1)
    reversed_image = np.roll(reversed_image, -(h // 2), axis=0)
    return reversed_image


def optimizeTileOrigin(tile):
    h, w = tile.shape[:2]
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    color = tile.astype(np.float32)
    vertical_diff = np.mean(np.abs(color - np.roll(color, 1, axis=1)), axis=(0, 2))
    horizontal_diff = np.mean(np.abs(color - np.roll(color, 1, axis=0)), axis=(1, 2))

    vertical_structure = np.mean(grad_x + grad_y, axis=0)
    horizontal_structure = np.mean(grad_x + grad_y, axis=1)

    vertical_score = vertical_diff + vertical_structure * 0.08
    horizontal_score = horizontal_diff + horizontal_structure * 0.08

    smooth_window = _odd_kernel(max(5, min(h, w) // 64))
    vertical_score = cv2.GaussianBlur(vertical_score.reshape(1, -1), (smooth_window, 1), 0).ravel()
    horizontal_score = cv2.GaussianBlur(horizontal_score.reshape(-1, 1), (1, smooth_window), 0).ravel()

    best_x = int(np.argmin(vertical_score))
    best_y = int(np.argmin(horizontal_score))

    shifted = np.roll(tile, -best_x, axis=1)
    shifted = np.roll(shifted, -best_y, axis=0)
    return shifted, {
        "origin_shift_x": best_x,
        "origin_shift_y": best_y,
        "origin_vertical_score": round(float(vertical_score[best_x]), 3),
        "origin_horizontal_score": round(float(horizontal_score[best_y]), 3),
    }


def wrapEdgeHarmonize(tile, band):
    h, w = tile.shape[:2]
    min_dim = min(h, w)
    edge_band = int(np.clip(max(band * 2, min_dim // 44), 8, max(8, min_dim // 18)))
    result = tile.astype(np.float32)
    ramp = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, edge_band, dtype=np.float32)))
    left_alpha = ramp.reshape(1, edge_band, 1)
    right_alpha = ramp[::-1].reshape(1, edge_band, 1)
    top_alpha = ramp.reshape(edge_band, 1, 1)
    bottom_alpha = ramp[::-1].reshape(edge_band, 1, 1)

    left = result[:, :edge_band].copy()
    right = result[:, -edge_band:].copy()
    right_from_left = np.flip(left, axis=1)
    left_from_right = np.flip(right, axis=1)

    left_transition = float(np.mean(np.abs(left_from_right[:, -1] - result[:, min(edge_band, w - 1)])))
    right_transition = float(np.mean(np.abs(right_from_left[:, 0] - result[:, max(0, w - edge_band - 1)])))
    if left_transition <= right_transition:
        result[:, :edge_band] = result[:, :edge_band] * (1.0 - left_alpha) + left_from_right * left_alpha
    else:
        result[:, -edge_band:] = result[:, -edge_band:] * (1.0 - right_alpha) + right_from_left * right_alpha

    top = result[:edge_band, :].copy()
    bottom = result[-edge_band:, :].copy()
    bottom_from_top = np.flip(top, axis=0)
    top_from_bottom = np.flip(bottom, axis=0)

    top_transition = float(np.mean(np.abs(top_from_bottom[-1, :] - result[min(edge_band, h - 1), :])))
    bottom_transition = float(np.mean(np.abs(bottom_from_top[0, :] - result[max(0, h - edge_band - 1), :])))
    if top_transition <= bottom_transition:
        result[:edge_band, :] = result[:edge_band, :] * (1.0 - top_alpha) + top_from_bottom * top_alpha
    else:
        result[-edge_band:, :] = result[-edge_band:, :] * (1.0 - bottom_alpha) + bottom_from_top * bottom_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def copyCompleteBorders(tile, band):
    h, w = tile.shape[:2]
    width = int(np.clip(max(band, min(h, w) // 56), 8, max(8, min(h, w) // 24)))
    result = tile.astype(np.float32)

    ramp = np.linspace(1.0, 0.0, width, dtype=np.float32)
    left_alpha = ramp.reshape(1, width, 1)
    right_alpha = ramp[::-1].reshape(1, width, 1)
    top_alpha = ramp.reshape(width, 1, 1)
    bottom_alpha = ramp[::-1].reshape(width, 1, 1)

    left = result[:, :width].copy()
    right = result[:, -width:].copy()

    left_transition = float(np.mean(np.abs(right[:, -1:] - result[:, width:width + 1])))
    right_transition = float(np.mean(np.abs(left[:, :1] - result[:, w - width - 1:w - width])))
    if left_transition <= right_transition:
        result[:, :width] = result[:, :width] * (1.0 - left_alpha) + right * left_alpha
    else:
        result[:, -width:] = result[:, -width:] * (1.0 - right_alpha) + left * right_alpha

    top = result[:width, :].copy()
    bottom = result[-width:, :].copy()

    top_transition = float(np.mean(np.abs(bottom[-1:, :] - result[width:width + 1, :])))
    bottom_transition = float(np.mean(np.abs(top[:1, :] - result[h - width - 1:h - width, :])))
    if top_transition <= bottom_transition:
        result[:width, :] = result[:width, :] * (1.0 - top_alpha) + bottom * top_alpha
    else:
        result[-width:, :] = result[-width:, :] * (1.0 - bottom_alpha) + top * bottom_alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def lockTileBorders(tile, width):
    h, w = tile.shape[:2]
    width = int(np.clip(width, 1, max(1, min(h, w) // 80)))
    result = tile.astype(np.float32)

    ramp = np.linspace(1.0, 0.0, width, dtype=np.float32)
    for i in range(width):
        alpha = ramp[i]

        left = result[:, i:i + 1].copy()
        right = result[:, w - i - 1:w - i].copy()
        shared_vertical = (left + right) * 0.5
        result[:, i:i + 1] = result[:, i:i + 1] * (1.0 - alpha) + shared_vertical * alpha
        result[:, w - i - 1:w - i] = result[:, w - i - 1:w - i] * (1.0 - alpha) + shared_vertical * alpha

        top = result[i:i + 1, :].copy()
        bottom = result[h - i - 1:h - i, :].copy()
        shared_horizontal = (top + bottom) * 0.5
        result[i:i + 1, :] = result[i:i + 1, :] * (1.0 - alpha) + shared_horizontal * alpha
        result[h - i - 1:h - i, :] = result[h - i - 1:h - i, :] * (1.0 - alpha) + shared_horizontal * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def validateTile3x3(tile, preview_path=None, max_preview_side=1800):
    preview = np.tile(tile, (3, 3, 1))
    export_preview = preview

    max_side = max(preview.shape[:2])
    if max_side > max_preview_side:
        scale = max_preview_side / max_side
        export_preview = cv2.resize(
            preview,
            (int(preview.shape[1] * scale), int(preview.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )

    if preview_path is not None:
        cv2.imwrite(preview_path, export_preview)

    left_right_error = float(np.mean(np.abs(tile[:, 0].astype(np.float32) - tile[:, -1].astype(np.float32))))
    top_bottom_error = float(np.mean(np.abs(tile[0, :].astype(np.float32) - tile[-1, :].astype(np.float32))))
    h, w = tile.shape[:2]
    edge_band = int(np.clip(min(h, w) // 32, 8, 64))
    left_band = tile[:, :edge_band].astype(np.float32)
    right_band = np.flip(tile[:, -edge_band:].astype(np.float32), axis=1)
    top_band = tile[:edge_band, :].astype(np.float32)
    bottom_band = np.flip(tile[-edge_band:, :].astype(np.float32), axis=0)
    left_right_band_error = float(np.mean(np.abs(left_band - right_band)))
    top_bottom_band_error = float(np.mean(np.abs(top_band - bottom_band)))

    return {
        "left_right_error": round(left_right_error, 3),
        "top_bottom_error": round(top_bottom_error, 3),
        "left_right_band_error": round(left_right_band_error, 3),
        "top_bottom_band_error": round(top_bottom_band_error, 3),
        "edge_band": edge_band,
        "preview_shape": list(preview.shape),
        "exported_preview_shape": list(export_preview.shape),
    }


def _quality_flags(validation):
    band_error = max(
        validation["left_right_band_error"],
        validation["top_bottom_band_error"],
    )
    pixel_error = max(
        validation["left_right_error"],
        validation["top_bottom_error"],
    )

    repeat_ready = pixel_error <= 2.5 and band_error <= 8.0
    needs_ai_inpaint = band_error > 8.0

    if repeat_ready:
        rating = "good"
    elif pixel_error <= 5.0 and band_error <= 14.0:
        rating = "usable"
    else:
        rating = "needs_review"

    return {
        "quality_rating": rating,
        "quality_score": round(float(pixel_error * 0.25 + band_error * 0.75), 3),
        "repeat_ready": bool(repeat_ready),
        "needs_ai_inpaint": bool(needs_ai_inpaint),
        "visible_seam_reason": (
            "Opposite edge bands do not naturally continue; generative seam inpainting is required."
            if needs_ai_inpaint else None
        ),
    }


def _candidate_score(tile):
    validation = validateTile3x3(tile)
    return validation, _quality_flags(validation)["quality_score"]


def _vertex_repair_multipliers():
    raw = os.getenv("VERTEX_AI_REPAIR_MULTIPLIERS", "1.25,2.0,3.0")
    values = []
    for part in raw.split(","):
        try:
            value = float(part.strip())
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values or [1.25, 2.0, 3.0]


def _try_wrap_edge_harmonize(tile, metadata):
    validation, current_score = _candidate_score(tile)
    flags = _quality_flags(validation)
    if not flags["needs_ai_inpaint"]:
        return tile, {
            "edge_harmonize_status": "not_needed",
            "edge_harmonize_before_score": round(float(current_score), 3),
        }

    band = int(metadata.get("repair_band") or metadata.get("seam_band") or validation["edge_band"])
    candidate = wrapEdgeHarmonize(tile, band)
    candidate_validation, candidate_score = _candidate_score(candidate)
    candidate_flags = _quality_flags(candidate_validation)

    if candidate_score + 0.5 < current_score:
        return candidate, {
            "edge_harmonize_status": "used",
            "edge_harmonize_before_score": round(float(current_score), 3),
            "edge_harmonize_after_score": round(float(candidate_score), 3),
            "edge_harmonize_before_band_error": round(float(max(
                validation["left_right_band_error"],
                validation["top_bottom_band_error"],
            )), 3),
            "edge_harmonize_after_band_error": round(float(max(
                candidate_validation["left_right_band_error"],
                candidate_validation["top_bottom_band_error"],
            )), 3),
            "edge_harmonize_quality": candidate_flags["quality_rating"],
        }

    return tile, {
        "edge_harmonize_status": "skipped_no_improvement",
        "edge_harmonize_before_score": round(float(current_score), 3),
        "edge_harmonize_candidate_score": round(float(candidate_score), 3),
    }


def _make_texture_quilt_tile(image):
    offset = offsetTransform(image)
    mask, band = seamMaskDetection(offset)
    quilted = textureQuiltSeam(offset, mask, band)
    result = reverseOffset(quilted)
    result, origin = optimizeTileOrigin(result)
    result = lockTileBorders(result, max(1, band // 6))

    metadata = {
        "method": "texture_quilt",
        "seam_band": band,
    }
    metadata.update(origin)
    return result, metadata


def _make_patch_transfer_tile(image):
    offset = offsetTransform(image)
    mask, band = seamMaskDetection(offset)
    foreground = _foreground_mask(offset, band)
    repair_band = int(np.clip(band * 2.0, band + 6, max(band + 6, min(image.shape[:2]) // 8)))
    seam_core = cv2.dilate(mask, np.ones((_odd_kernel(repair_band), _odd_kernel(repair_band)), dtype=np.uint8), iterations=1)
    seam_fragments = _seam_fragment_mask(seam_core, foreground, repair_band)
    repair_mask = cv2.bitwise_or(seam_core, seam_fragments)

    repaired = patchTransferSeam(offset, repair_mask, repair_band)
    result = reverseOffset(repaired)
    result, origin = optimizeTileOrigin(result)
    result = lockTileBorders(result, 1)

    metadata = {
        "method": "patch_transfer",
        "seam_band": band,
        "repair_band": repair_band,
        "patch_repair_ratio": round(float(np.mean(repair_mask > 0)), 4),
    }
    metadata.update(origin)
    return result, metadata


def _make_structure_preserving_tile(image, cleanup_scale=1.0):
    offset = offsetTransform(image)
    mask, band = seamMaskDetection(offset)
    repair_band = int(max(band, round(band * cleanup_scale)))

    foreground = _foreground_mask(offset, band)
    if cleanup_scale > 1.0:
        widen_kernel = np.ones((_odd_kernel(repair_band), _odd_kernel(repair_band)), dtype=np.uint8)
        seam_core = cv2.dilate(mask, widen_kernel, iterations=1)
    else:
        seam_core = mask

    seam_fragments = _seam_fragment_mask(seam_core, foreground, repair_band)
    repair_mask = cv2.bitwise_or(seam_core, seam_fragments)

    repaired = offset.copy()
    if np.count_nonzero(repair_mask) > 0:
        inpainted = opencvInpainting(offset, repair_mask, radius=int(np.clip(repair_band // 2, 3, 9)))
        repaired = _masked_blend(offset, inpainted, repair_mask, repair_band, strength=1.0)
        repaired = edgeCompletion(repaired, repair_mask, max(4, repair_band // 2))
        repaired = lumiStats(repaired, offset, repair_mask, max(4, repair_band // 2))

    result = reverseOffset(repaired)
    result = lockTileBorders(result, 1)

    metadata = {
        "method": "structure_preserving" if cleanup_scale <= 1.0 else "structure_preserving_aggressive",
        "seam_band": band,
        "repair_band": repair_band,
        "foreground_protected_ratio": round(float(np.mean(foreground > 0)), 4),
        "seam_fragment_ratio": round(float(np.count_nonzero(seam_fragments) / max(1, np.count_nonzero(mask))), 4),
        "repaired_seam_ratio": round(float(np.count_nonzero(repair_mask) / max(1, np.count_nonzero(mask))), 4),
        "origin_shift_x": 0,
        "origin_shift_y": 0,
        "origin_vertical_score": None,
        "origin_horizontal_score": None,
    }
    return result, metadata


def _make_vertex_inpaint_tile(image):
    offset = offsetTransform(image)
    mask, band = seamMaskDetection(offset)
    base_metadata = {
        "method": "vertex_imagen_inpaint",
        "seam_band": band,
    }

    best_result = None
    best_validation = None
    best_score = None
    best_metadata = {}
    attempt_metadata = []

    for multiplier in _vertex_repair_multipliers():
        repair_band = int(np.clip(
            round(band * multiplier),
            band + 6,
            max(band + 6, min(image.shape[:2]) // 7),
        ))
        widen_kernel = np.ones((_odd_kernel(repair_band), _odd_kernel(repair_band)), dtype=np.uint8)
        edit_mask = cv2.dilate(mask, widen_kernel, iterations=1)

        edited_offsets, vertex_meta = _vertex_edit_offset(offset, edit_mask)
        attempt = {
            "repair_multiplier": round(float(multiplier), 3),
            "repair_band": repair_band,
            "vertex_edit_ratio": round(float(np.mean(edit_mask > 0)), 4),
            **vertex_meta,
        }

        if edited_offsets is None:
            attempt_metadata.append(attempt)
            continue

        for index, edited_offset in enumerate(edited_offsets):
            composited_offset = _masked_blend(offset, edited_offset, edit_mask, repair_band, strength=1.0)
            candidate = reverseOffset(composited_offset)
            candidate = lockTileBorders(candidate, 1)
            candidate_validation, candidate_score = _candidate_score(candidate)
            candidate_flags = _quality_flags(candidate_validation)

            if best_score is None or candidate_score < best_score:
                best_result = candidate
                best_validation = candidate_validation
                best_score = candidate_score
                best_metadata = {
                    **attempt,
                    "repair_band": repair_band,
                    "vertex_selected_candidate": index,
                    "vertex_best_score": round(float(best_score), 3),
                    "vertex_best_left_right_band_error": candidate_validation["left_right_band_error"],
                    "vertex_best_top_bottom_band_error": candidate_validation["top_bottom_band_error"],
                    "vertex_best_quality": candidate_flags["quality_rating"],
                }

            if candidate_flags["repeat_ready"]:
                base_metadata.update(best_metadata)
                base_metadata["vertex_attempt_count"] = len(attempt_metadata) + 1
                base_metadata["origin_shift_x"] = 0
                base_metadata["origin_shift_y"] = 0
                base_metadata["origin_vertical_score"] = None
                base_metadata["origin_horizontal_score"] = None
                return best_result, base_metadata

        attempt["vertex_best_attempt_score"] = (
            round(float(best_score), 3) if best_score is not None else None
        )
        attempt_metadata.append(attempt)

    if best_result is None:
        return None, {
            **base_metadata,
            "vertex_status": "no_candidate_selected",
            "vertex_attempts": attempt_metadata,
        }

    base_metadata.update({
        **best_metadata,
        "origin_shift_x": 0,
        "origin_shift_y": 0,
        "origin_vertical_score": None,
        "origin_horizontal_score": None,
        "vertex_attempt_count": len(attempt_metadata),
    })
    return best_result, base_metadata


def make_seamless(input_path, output_path, preview_path=None):
    image = cv2.imread(input_path)

    if image is None:
        raise Exception("Unable to load image")

    _, profile_band = seamMaskDetection(offsetTransform(image))
    profile = _texture_profile(image, profile_band)

    result = None
    metadata = {}
    ai_required = os.getenv("VERTEX_AI_INPAINT_REQUIRED", "1") != "0"

    # 1. Try Vertex AI Imagen path first if enabled and the SDK is installed.
    if os.getenv("VERTEX_AI_INPAINT_ENABLED", "1") != "0":
        try:
            vertex_result, vertex_metadata = _make_vertex_inpaint_tile(image)
            if vertex_result is not None:
                result = vertex_result
                metadata = vertex_metadata
            else:
                metadata.update({
                    k: v for k, v in vertex_metadata.items()
                    if k.startswith("vertex_") or k == "ai_provider"
                })
        except Exception as exc:
            metadata.update({
                "ai_provider": "vertex_ai_imagen",
                "vertex_status": "failed",
                "vertex_error": str(exc)[:240],
            })
    elif ai_required:
        raise Exception("Vertex AI seam inpainting is required but VERTEX_AI_INPAINT_ENABLED is disabled.")

    if result is None and ai_required:
        detail = metadata.get("vertex_error") or metadata.get("vertex_status") or "unknown error"
        raise Exception(
            "AI seam inpainting failed, so no local fallback image was returned. "
            f"Vertex AI status: {metadata.get('vertex_status', 'not_used')}. Detail: {detail}"
        )

    # 2. Fall back to local algorithms if Vertex AI is disabled, not configured, or failed.
    if result is None:
        selected_metadata = {}
        if profile["is_structured"]:
            # Structure preserving algorithm
            result, metadata_local = _make_structure_preserving_tile(image)
            selected_metadata = metadata_local
            
            # Check if we should try aggressive cleanup
            quick_validation, current_score = _candidate_score(result)
            if max(quick_validation["left_right_band_error"], quick_validation["top_bottom_band_error"]) > 12.0:
                aggressive_result, aggressive_metadata = _make_structure_preserving_tile(image, cleanup_scale=1.65)
                aggressive_validation, aggressive_score = _candidate_score(aggressive_result)
                if aggressive_score <= current_score:
                    result = aggressive_result
                    selected_metadata = aggressive_metadata
                    quick_validation = aggressive_validation
                    current_score = aggressive_score

            # Floral and wallpaper-style artwork often gets visible locked-edge bands
            # from the structure-preserving path. Keep a texture-quilt candidate and
            # choose it when validation says the repeated band is cleaner.
            current_band_error = max(
                quick_validation["left_right_band_error"],
                quick_validation["top_bottom_band_error"],
            )
            if current_band_error > 10.0 or current_score > 8.0:
                texture_result, texture_metadata = _make_texture_quilt_tile(image)
                _, texture_score = _candidate_score(texture_result)
                if texture_score + 0.75 < current_score:
                    result = texture_result
                    selected_metadata = texture_metadata
                    current_score = texture_score
        else:
            # Texture quilting algorithm
            result, metadata_local = _make_texture_quilt_tile(image)
            selected_metadata = metadata_local

        metadata.update(selected_metadata)

    # 3. Perform final validation, preview generation, and save image
    validation = validateTile3x3(result, preview_path)
    validation.update(profile)
    validation.update(metadata)
    validation.update(_quality_flags(validation))

    strict_repeat_required = os.getenv("STRICT_REPEAT_REQUIRED", "0") != "0"
    if ai_required and validation["needs_ai_inpaint"] and strict_repeat_required:
        raise Exception(
            "AI seam inpainting ran but the output failed strict repeat validation. "
            f"left_right_band_error={validation['left_right_band_error']}, "
            f"top_bottom_band_error={validation['top_bottom_band_error']}, "
            f"vertex_status={validation.get('vertex_status', 'unknown')}, "
            f"repair_band={validation.get('repair_band', 'unknown')}, "
            f"candidate={validation.get('vertex_selected_candidate', 'unknown')}, "
            f"best_score={validation.get('vertex_best_score', 'unknown')}"
        )

    if ai_required and validation["needs_ai_inpaint"]:
        validation["strict_validation_status"] = "failed_non_blocking"
        validation["visible_seam_reason"] = (
            "AI output was generated, but strict repeat validation still detected visible seam risk. "
            f"left/right band error {validation['left_right_band_error']}, "
            f"top/bottom band error {validation['top_bottom_band_error']}."
        )
    else:
        validation["strict_validation_status"] = "passed"

    cv2.imwrite(output_path, result)

    return {
        "output_path": output_path,
        "preview_path": preview_path,
        "validation": validation,
    }
