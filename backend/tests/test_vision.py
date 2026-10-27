"""Smoke tests for the 3-tier OCR pipeline.

These tests avoid external services: Azure CV is monkeypatched, and
Tesseract tests are skipped when the binary is absent so the suite still runs
on bare CI boxes.
"""
from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag import vision  # noqa: E402
from app.rag.vision import (  # noqa: E402
    OcrResult, OcrTier, enrich_text_with_ocr, extract_proximity_caption,
    ocr_image_region, ocr_page_scanned, ocr_pdf_document, preprocess_image,
)


class _StubPage:
    def __init__(self, page_num: int, text: str, bbox_image_regions: list[tuple[float, float, float, float]]):
        self.page_num = page_num
        self.text = text
        self.bbox_image_regions = bbox_image_regions

HAS_TESSERACT = shutil.which("tesseract") is not None


def _pdf_with_caption() -> bytes:
    """PDF with a visible image region and a caption nearby."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    # Draw a rectangle as a stand-in for an image.
    rect = fitz.Rect(50, 50, 250, 200)
    page.draw_rect(rect, color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    # Caption directly below.
    page.insert_text((50, 220), "Figure 1: Quarterly revenue by region, 2026", fontsize=11)
    page.insert_text((50, 240), "Source: internal finance dashboard, screenshot taken 2026-10-01", fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _pdf_blank_scanned_page() -> bytes:
    """Scanned-style page: embed a PNG, no page text, nothing nearby."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    img = Image.new("RGB", (200, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(fitz.Rect(100, 150, 300, 250), stream=buf.getvalue())
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_preprocess_image_returns_grayscale():
    img = Image.new("RGB", (120, 80), "white")
    out = preprocess_image(img)
    assert out.mode == "L"
    assert out.size == (120, 80)


def test_proximity_caption_finds_nearby_text():
    pdf_bytes = _pdf_with_caption()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[0]
        text, conf = extract_proximity_caption(page, fitz.Rect(50, 50, 250, 200))
    assert "Figure 1" in text
    assert conf > 0.0


def test_tier_1_accepts_proximity_caption():
    pdf_bytes = _pdf_with_caption()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[0]
        result = ocr_image_region(
            page, fitz.Rect(50, 50, 250, 200),
            page_num=1, enable_azure=False,
        )
    assert isinstance(result, OcrResult)
    assert result.tier == OcrTier.PROXIMITY
    assert "Figure 1" in result.text
    assert result.tier_attempts == [OcrTier.PROXIMITY]


def test_tier_2_runs_when_proximity_empty(monkeypatch):
    """No nearby caption → Tier 1 fails the gate → Tesseract runs."""
    pdf_bytes = _pdf_blank_scanned_page()

    monkeypatch.setattr(
        vision, "run_tesseract",
        lambda img, lang="eng": ("Hello world sample text", 0.92),
    )
    monkeypatch.setattr(
        vision, "run_azure_cv",
        lambda img: ("SHOULD NOT RUN", 0.99, None),
    )
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[0]
        images = page.get_images(full=True)
        assert images, "test fixture must embed an image"
        bbox = page.get_image_rects(images[0][0])[0]
        result = ocr_image_region(page, bbox, page_num=1, enable_azure=True)

    assert result.tier == OcrTier.TESSERACT
    assert result.text == "Hello world sample text"
    assert OcrTier.AZURE_CV not in result.tier_attempts


def test_tier_3_runs_when_tesseract_low_conf(monkeypatch):
    """Tesseract succeeds but at low confidence → fall through to Azure CV."""
    pdf_bytes = _pdf_blank_scanned_page()

    monkeypatch.setattr(vision, "run_tesseract", lambda img, lang="eng": ("hi", 0.20))
    monkeypatch.setattr(
        vision, "run_azure_cv",
        lambda img: ("Handwritten note: pay invoice #4421", 0.88, None),
    )
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[0]
        bbox = page.get_image_rects(page.get_images(full=True)[0][0])[0]
        result = ocr_image_region(page, bbox, page_num=1, enable_azure=True)

    assert result.tier == OcrTier.AZURE_CV
    assert "invoice" in result.text
    assert result.tier_attempts == [OcrTier.PROXIMITY, OcrTier.TESSERACT, OcrTier.AZURE_CV]


def test_azure_disabled_falls_back_to_best(monkeypatch):
    pdf_bytes = _pdf_blank_scanned_page()
    monkeypatch.setattr(vision, "run_tesseract", lambda img, lang="eng": ("blurry", 0.15))
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[0]
        bbox = page.get_image_rects(page.get_images(full=True)[0][0])[0]
        result = ocr_image_region(page, bbox, page_num=1, enable_azure=False)
    assert result.tier == OcrTier.TESSERACT
    assert result.text == "blurry"
    assert OcrTier.AZURE_CV not in result.tier_attempts


def test_ocr_pdf_document_walks_pages(monkeypatch):
    pdf_bytes = _pdf_with_caption()
    monkeypatch.setattr(vision, "run_tesseract", lambda img, lang="eng": ("", 0.0))
    monkeypatch.setattr(vision, "run_azure_cv", lambda img: ("", 0.0, "disabled"))
    results = ocr_pdf_document(pdf_bytes, enable_azure=False)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, OcrResult)
        assert r.page_num >= 1


def test_azure_rejects_tiny_image(monkeypatch):
    class _StubSettings:
        azure_ai_api_key = "fake-key"
        azure_ai_api_base = "https://example.invalid"
    monkeypatch.setattr(vision, "get_settings", lambda: _StubSettings())
    tiny = Image.new("RGB", (10, 10), "white")
    text, conf, err = vision.run_azure_cv(tiny)
    assert text == ""
    assert conf == 0.0
    assert err is not None and "50px" in err


def test_azure_no_key_returns_error(monkeypatch):
    class _FakeSettings:
        azure_ai_api_key = ""
        azure_ai_api_base = "https://example.invalid"
    monkeypatch.setattr(vision, "get_settings", lambda: _FakeSettings())
    img = Image.new("RGB", (100, 100), "white")
    text, conf, err = vision.run_azure_cv(img)
    assert text == ""
    assert conf == 0.0
    assert "AZURE_AI_API_KEY" in (err or "")


def test_enrich_text_appends_ocr_output(monkeypatch):
    pdf_bytes = _pdf_with_caption()
    pages = [_StubPage(page_num=1, text="native page text",
                       bbox_image_regions=[(50.0, 50.0, 250.0, 200.0)])]
    monkeypatch.setattr(vision, "run_tesseract", lambda img, lang="eng": ("", 0.0))
    enriched, dominant_tier = enrich_text_with_ocr(pdf_bytes, pages, enable_azure=False)
    assert len(enriched) == 1
    pnum, body = enriched[0]
    assert pnum == 1
    assert "native page text" in body
    # Proximity caption should be stitched in.
    assert "[OCR p1#0 tier=proximity" in body
    assert "Figure 1" in body
    assert dominant_tier == "proximity"


def test_enrich_text_handles_scanned_only_page(monkeypatch):
    pdf_bytes = _pdf_blank_scanned_page()
    # Simulate the extractor returning no text and no bbox metadata for this page
    # (older extractor paths): bridge should still fall back to full-page raster OCR.
    pages = [_StubPage(page_num=1, text="", bbox_image_regions=[])]
    monkeypatch.setattr(
        vision, "run_tesseract",
        lambda img, lang="eng": ("Scanned contents go here", 0.90),
    )
    monkeypatch.setattr(vision, "run_azure_cv", lambda img: ("UNUSED", 0.99, None))
    enriched, dominant_tier = enrich_text_with_ocr(pdf_bytes, pages, enable_azure=True)
    _, body = enriched[0]
    assert "Scanned contents go here" in body
    assert "full-page tier=tesseract" in body
    assert dominant_tier == "tesseract"


def test_enrich_text_filters_low_confidence(monkeypatch):
    pdf_bytes = _pdf_blank_scanned_page()
    pages = [_StubPage(page_num=1, text="", bbox_image_regions=[(100.0, 150.0, 300.0, 250.0)])]
    monkeypatch.setattr(vision, "run_tesseract", lambda img, lang="eng": ("junk", 0.05))
    enriched, dominant_tier = enrich_text_with_ocr(
        pdf_bytes, pages, enable_azure=False, confidence_floor=0.50,
    )
    _, body = enriched[0]
    # Low-conf output filtered → page body stays empty, no tier recorded.
    assert "junk" not in body
    assert dominant_tier is None


@pytest.mark.skipif(not HAS_TESSERACT, reason="tesseract binary not installed")
def test_tesseract_real_runs_on_rendered_text():
    img = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(img).text((20, 30), "HELLO 2026", fill="black")
    text, conf = vision.run_tesseract(img)
    # Tesseract output is noisy on PIL-rendered defaults; just ensure it ran.
    assert isinstance(text, str)
    assert 0.0 <= conf <= 1.0
