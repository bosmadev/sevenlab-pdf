"""T-005: 3-Tier OCR Pipeline.

Tier 1: Proximity captions   (PyMuPDF bbox neighbourhood, zero cost)
Tier 2: Tesseract OCR        (pytesseract, free, CPU)
Tier 3: Azure Computer Vision v4  (handwriting fallback, $1.50 / 1k images)

Downstream callers get a stable `OcrResult` regardless of which tier produced
the text. Confidence is 0.0-1.0 and comparable across tiers.
"""
from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pymupdf  # PyMuPDF 2.x — modern module name, `fitz` is the deprecated alias
import httpx
import pytesseract
from PIL import Image, ImageFilter, ImageOps

# Local alias so the rest of this file reads naturally.
fitz = pymupdf

from app.config import get_settings

log = logging.getLogger(__name__)

# Thresholds that trigger the next tier. Tuned conservatively — a scanned page
# with no proximity text drops to Tesseract; low-confidence Tesseract output
# (e.g. handwriting) drops to Azure CV v4.
PROXIMITY_MIN_CHARS = 20
PROXIMITY_MIN_CONF = 0.30
TESSERACT_MIN_CHARS = 15
TESSERACT_MIN_CONF = 0.55
AZURE_API_VERSION = "2024-02-01"
AZURE_IMAGE_ANALYSIS_PATH = "/computervision/imageanalysis:analyze"
AZURE_FEATURES = "read"  # OCR / handwriting
AZURE_TIMEOUT_S = 30.0
MIN_IMAGE_DIM = 50  # Azure rejects < 50x50


class OcrTier(str, Enum):
    PROXIMITY = "proximity"
    TESSERACT = "tesseract"
    AZURE_CV = "azure_cv"
    NONE = "none"


@dataclass
class OcrResult:
    """Unified OCR output across all three tiers."""
    text: str
    confidence: float
    tier: OcrTier
    page_num: int
    image_index: int = 0
    duration_ms: int = 0
    bbox: Optional[tuple[float, float, float, float]] = None
    error: Optional[str] = None
    tier_attempts: list[OcrTier] = field(default_factory=list)


def preprocess_image(img: Image.Image, *, deskew: bool = False, denoise: bool = True) -> Image.Image:
    """Grayscale + optional denoise/deskew to lift OCR accuracy on scans."""
    if img.mode != "L":
        img = img.convert("L")
    if denoise:
        img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img, cutoff=1)
    if deskew:
        # Light deskew via PIL only. A full Hough-based deskew needs OpenCV,
        # which the MVP skips to keep the dependency surface small.
        pass
    return img


def extract_proximity_caption(
    page: "fitz.Page",
    image_bbox: fitz.Rect,
    *,
    radius: float = 80.0,
) -> tuple[str, float]:
    """Tier 1 — text blocks within `radius` points of an image bbox.

    Confidence is the ratio of neighbouring characters to a reference length,
    capped at 1.0. A page with lots of nearby prose scores higher than a page
    with a stray caption.
    """
    search_rect = fitz.Rect(
        image_bbox.x0 - radius,
        image_bbox.y0 - radius,
        image_bbox.x1 + radius,
        image_bbox.y1 + radius,
    )
    blocks = page.get_text("blocks", clip=search_rect) or []
    text_parts: list[str] = []
    for block in blocks:
        if len(block) >= 5 and isinstance(block[4], str):
            cleaned = block[4].strip()
            if cleaned:
                text_parts.append(cleaned)
    text = "\n".join(text_parts).strip()
    if not text:
        return "", 0.0
    confidence = min(1.0, len(text) / 200.0)
    return text, confidence


def _tesseract_confidence(data: dict) -> float:
    """Average word confidence from `image_to_data` output, 0.0-1.0."""
    confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
    if not confs:
        return 0.0
    return sum(confs) / len(confs) / 100.0


def run_tesseract(img: Image.Image, *, lang: str = "eng") -> tuple[str, float]:
    """Tier 2 — Tesseract OCR with per-word confidence averaging."""
    try:
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        log.error("tesseract binary not installed on PATH")
        return "", 0.0
    except Exception as exc:
        log.warning("tesseract failed: %s", exc)
        return "", 0.0
    words = [w for w in data.get("text", []) if w and w.strip()]
    text = " ".join(words).strip()
    confidence = _tesseract_confidence(data)
    return text, confidence


def _image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_azure_cv(img: Image.Image) -> tuple[str, float, Optional[str]]:
    """Tier 3 — Azure Computer Vision v4 Image Analysis (Read feature).

    Returns `(text, confidence, error)`. On error, text="" and confidence=0.0;
    `error` carries a short diagnostic string.
    """
    settings = get_settings()
    if not settings.azure_ai_api_key:
        return "", 0.0, "AZURE_AI_API_KEY not configured"
    if img.width < MIN_IMAGE_DIM or img.height < MIN_IMAGE_DIM:
        return "", 0.0, f"image below {MIN_IMAGE_DIM}px minimum"

    url = f"{settings.azure_ai_api_base.rstrip('/')}{AZURE_IMAGE_ANALYSIS_PATH}"
    params = {"api-version": AZURE_API_VERSION, "features": AZURE_FEATURES}
    headers = {
        "Ocp-Apim-Subscription-Key": settings.azure_ai_api_key,
        "Content-Type": "application/octet-stream",
    }
    payload = _image_to_png_bytes(img)

    try:
        with httpx.Client(timeout=AZURE_TIMEOUT_S) as client:
            resp = client.post(url, params=params, headers=headers, content=payload)
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPStatusError as exc:
        return "", 0.0, f"azure http {exc.response.status_code}"
    except httpx.HTTPError as exc:
        return "", 0.0, f"azure http error: {exc.__class__.__name__}"
    except Exception as exc:
        return "", 0.0, f"azure error: {exc}"

    read_result = body.get("readResult") or {}
    blocks = read_result.get("blocks") or []
    lines_text: list[str] = []
    confidences: list[float] = []
    for block in blocks:
        for line in block.get("lines", []) or []:
            txt = (line.get("text") or "").strip()
            if txt:
                lines_text.append(txt)
            for word in line.get("words", []) or []:
                c = word.get("confidence")
                if isinstance(c, (int, float)):
                    confidences.append(float(c))
    text = "\n".join(lines_text).strip()
    confidence = (sum(confidences) / len(confidences)) if confidences else (0.5 if text else 0.0)
    return text, confidence, None


def _pixmap_to_pil(pix: "fitz.Pixmap") -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)


def ocr_image_region(
    page: "fitz.Page",
    image_bbox: fitz.Rect,
    *,
    page_num: int,
    image_index: int = 0,
    enable_azure: bool = True,
) -> OcrResult:
    """Run tiers 1→2→3 against one image on a page, stopping on acceptance."""
    attempts: list[OcrTier] = []
    started = time.monotonic()

    # Tier 1
    attempts.append(OcrTier.PROXIMITY)
    text, conf = extract_proximity_caption(page, image_bbox)
    if len(text) >= PROXIMITY_MIN_CHARS and conf >= PROXIMITY_MIN_CONF:
        return OcrResult(
            text=text, confidence=conf, tier=OcrTier.PROXIMITY,
            page_num=page_num, image_index=image_index,
            duration_ms=int((time.monotonic() - started) * 1000),
            bbox=(image_bbox.x0, image_bbox.y0, image_bbox.x1, image_bbox.y1),
            tier_attempts=list(attempts),
        )

    # Rasterise the image bbox at 2x for OCR tiers.
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=image_bbox, alpha=False)
        img = _pixmap_to_pil(pix)
        img = preprocess_image(img)
    except Exception as exc:
        return OcrResult(
            text=text, confidence=conf, tier=OcrTier.NONE,
            page_num=page_num, image_index=image_index,
            duration_ms=int((time.monotonic() - started) * 1000),
            error=f"rasterise failed: {exc}",
            tier_attempts=list(attempts),
        )

    # Tier 2
    attempts.append(OcrTier.TESSERACT)
    t_text, t_conf = run_tesseract(img)
    if len(t_text) >= TESSERACT_MIN_CHARS and t_conf >= TESSERACT_MIN_CONF:
        return OcrResult(
            text=t_text, confidence=t_conf, tier=OcrTier.TESSERACT,
            page_num=page_num, image_index=image_index,
            duration_ms=int((time.monotonic() - started) * 1000),
            bbox=(image_bbox.x0, image_bbox.y0, image_bbox.x1, image_bbox.y1),
            tier_attempts=list(attempts),
        )

    # Tier 3 — Azure CV v4 (handwriting / complex layouts)
    if enable_azure:
        attempts.append(OcrTier.AZURE_CV)
        a_text, a_conf, a_err = run_azure_cv(img)
        if a_text:
            return OcrResult(
                text=a_text, confidence=a_conf, tier=OcrTier.AZURE_CV,
                page_num=page_num, image_index=image_index,
                duration_ms=int((time.monotonic() - started) * 1000),
                bbox=(image_bbox.x0, image_bbox.y0, image_bbox.x1, image_bbox.y1),
                error=a_err,
                tier_attempts=list(attempts),
            )
        # Keep the Azure error for diagnostics even on fallback.
        return _best_fallback(
            attempts, started, page_num, image_index, image_bbox,
            (text, conf, OcrTier.PROXIMITY),
            (t_text, t_conf, OcrTier.TESSERACT),
            azure_error=a_err,
        )

    return _best_fallback(
        attempts, started, page_num, image_index, image_bbox,
        (text, conf, OcrTier.PROXIMITY),
        (t_text, t_conf, OcrTier.TESSERACT),
    )


def _best_fallback(
    attempts: list[OcrTier],
    started: float,
    page_num: int,
    image_index: int,
    bbox: fitz.Rect,
    *candidates: tuple[str, float, OcrTier],
    azure_error: Optional[str] = None,
) -> OcrResult:
    """Pick the highest-confidence tier output when nothing cleared the gate."""
    best = max(candidates, key=lambda c: c[1]) if candidates else ("", 0.0, OcrTier.NONE)
    text, conf, tier = best
    return OcrResult(
        text=text, confidence=conf, tier=tier if text else OcrTier.NONE,
        page_num=page_num, image_index=image_index,
        duration_ms=int((time.monotonic() - started) * 1000),
        bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
        error=azure_error,
        tier_attempts=list(attempts),
    )


def ocr_page_scanned(
    page: "fitz.Page",
    *,
    page_num: int,
    enable_azure: bool = True,
) -> OcrResult:
    """For a page with NO embedded text — rasterise the whole page and OCR.

    Skips Tier 1 (there is no proximity text to find) and goes straight
    Tesseract → Azure CV.
    """
    started = time.monotonic()
    attempts: list[OcrTier] = []
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = preprocess_image(_pixmap_to_pil(pix))
    except Exception as exc:
        return OcrResult(
            text="", confidence=0.0, tier=OcrTier.NONE,
            page_num=page_num, duration_ms=int((time.monotonic() - started) * 1000),
            error=f"rasterise failed: {exc}",
            tier_attempts=list(attempts),
        )

    attempts.append(OcrTier.TESSERACT)
    t_text, t_conf = run_tesseract(img)
    if len(t_text) >= TESSERACT_MIN_CHARS and t_conf >= TESSERACT_MIN_CONF:
        return OcrResult(
            text=t_text, confidence=t_conf, tier=OcrTier.TESSERACT,
            page_num=page_num, duration_ms=int((time.monotonic() - started) * 1000),
            tier_attempts=list(attempts),
        )

    if enable_azure:
        attempts.append(OcrTier.AZURE_CV)
        a_text, a_conf, a_err = run_azure_cv(img)
        if a_text:
            return OcrResult(
                text=a_text, confidence=a_conf, tier=OcrTier.AZURE_CV,
                page_num=page_num, duration_ms=int((time.monotonic() - started) * 1000),
                error=a_err,
                tier_attempts=list(attempts),
            )

    # Low-confidence Tesseract output is still better than nothing.
    tier = OcrTier.TESSERACT if t_text else OcrTier.NONE
    return OcrResult(
        text=t_text, confidence=t_conf, tier=tier,
        page_num=page_num, duration_ms=int((time.monotonic() - started) * 1000),
        tier_attempts=list(attempts),
    )


def enrich_text_with_ocr(
    pdf_bytes: bytes,
    pages: list,
    *,
    enable_azure: bool = True,
    confidence_floor: float = 0.30,
) -> tuple[list[tuple[int, str]], Optional[str]]:
    """Bridge for the parsing→chunking pipeline (T-005 integration seam).

    Given the raw PDF bytes and a list of extractor pages (any object with
    `.page_num`, `.text`, `.bbox_image_regions`), OCR every image bbox,
    plus whole-page raster when a page has no text and no bboxes. Appended
    OCR text is tagged `[OCR p<page>#<idx> tier=<t> conf=<f>]` so citations
    downstream can distinguish OCR from native text.

    Returns `(enriched, dominant_tier)` where:
      enriched: list of (page_num, enriched_text) in page order
      dominant_tier: the tier that produced the most OCR characters, or None
                     if no OCR fired. Suitable for `documents.ocr_tier_used`.
    """
    enriched: list[tuple[int, str]] = []
    chars_per_tier: dict[str, int] = {}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_obj in pages:
            page_num = getattr(page_obj, "page_num", 0)
            base_text = getattr(page_obj, "text", "") or ""
            bboxes = getattr(page_obj, "bbox_image_regions", []) or []
            if page_num < 1 or page_num > len(doc):
                enriched.append((page_num, base_text))
                continue

            page = doc[page_num - 1]
            ocr_chunks: list[str] = []

            for image_index, bbox_tuple in enumerate(bboxes):
                try:
                    bbox = fitz.Rect(*bbox_tuple)
                except Exception:
                    continue
                result = ocr_image_region(
                    page, bbox,
                    page_num=page_num, image_index=image_index,
                    enable_azure=enable_azure,
                )
                if result.text and result.confidence >= confidence_floor:
                    ocr_chunks.append(
                        f"[OCR p{page_num}#{image_index} tier={result.tier.value} "
                        f"conf={result.confidence:.2f}]\n{result.text}"
                    )
                    chars_per_tier[result.tier.value] = (
                        chars_per_tier.get(result.tier.value, 0) + len(result.text)
                    )

            if not base_text.strip() and not bboxes:
                result = ocr_page_scanned(page, page_num=page_num, enable_azure=enable_azure)
                if result.text and result.confidence >= confidence_floor:
                    ocr_chunks.append(
                        f"[OCR p{page_num} full-page tier={result.tier.value} "
                        f"conf={result.confidence:.2f}]\n{result.text}"
                    )
                    chars_per_tier[result.tier.value] = (
                        chars_per_tier.get(result.tier.value, 0) + len(result.text)
                    )

            if ocr_chunks:
                combined = (base_text + "\n\n" + "\n\n".join(ocr_chunks)).strip()
            else:
                combined = base_text
            enriched.append((page_num, combined))

    dominant_tier = max(chars_per_tier, key=chars_per_tier.get) if chars_per_tier else None
    return enriched, dominant_tier


def ocr_pdf_document(pdf_bytes: bytes, *, enable_azure: bool = True) -> list[OcrResult]:
    """Walk every page: OCR each embedded image; if a page has no text at all,
    OCR the full page raster. Returns one `OcrResult` per OCR event."""
    results: list[OcrResult] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            try:
                images = page.get_images(full=True) or []
                page_text = (page.get_text("text") or "").strip()
            except Exception as exc:
                log.warning("page %d: could not inspect: %s", page_num, exc)
                continue

            for image_index, img_info in enumerate(images):
                xref = img_info[0]
                try:
                    rects = page.get_image_rects(xref) or []
                except Exception:
                    rects = []
                bbox = rects[0] if rects else page.rect
                results.append(ocr_image_region(
                    page, bbox,
                    page_num=page_num,
                    image_index=image_index,
                    enable_azure=enable_azure,
                ))

            if not page_text and not images:
                # Fully scanned page with no embedded images AND no text.
                results.append(ocr_page_scanned(page, page_num=page_num, enable_azure=enable_azure))
    return results
