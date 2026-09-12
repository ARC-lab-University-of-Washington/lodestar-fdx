# AI use statement: docs/AI_USE.md
##
# @file pdf_ingest.py
# @brief PDF -> raw segments, split on procedure/section structure rather than fixed-size windows, so a segment stays a coherent unit (one procedure step, one .
#
# malfunction entry) — which is what both citation and graph construction
# depend on. Every page is OCR'd fresh via Tesseract rather than trusting
# pdfplumber's embedded-text-layer extraction. This corpus is Apollo-era
# scans: many pages have no text layer at all (extract_text() silently returns
# nothing, and the page used to get dropped with zero warning), and where a
# text layer *does* exist it's often a low-quality legacy OCR pass already
# baked into the scan (garbled tokens like "MALFUi_CTION"). For a citation-
# grounded, safety-critical corpus we want one known, confidence-scored
# transcription path rather than silently inheriting whatever (if anything)
# the source PDF happens to carry.
#
"""
PDF -> raw segments, split on procedure/section structure rather than
fixed-size windows, so a segment stays a coherent unit (one procedure
step, one malfunction entry) — which is what both citation and graph
construction depend on.

Every page is OCR'd fresh via Tesseract rather than trusting pdfplumber's
embedded-text-layer extraction. This corpus is Apollo-era scans: many pages
have no text layer at all (extract_text() silently returns nothing, and the
page used to get dropped with zero warning), and where a text layer *does*
exist it's often a low-quality legacy OCR pass already baked into the scan
(garbled tokens like "MALFUi_CTION"). For a citation-grounded, safety-critical
corpus we want one known, confidence-scored transcription path rather than
silently inheriting whatever (if anything) the source PDF happens to carry.
"""
import os
import re
import shutil
import pdfplumber
import pytesseract
from concurrent.futures import ProcessPoolExecutor

# Locate the Tesseract engine binary robustly: honor $TESSERACT_CMD, else take
# whatever is on PATH, else fall back to the default Windows install location.
# Prevents TesseractNotFoundError when PATH hasn't been refreshed in this process.
_tcmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
if not _tcmd and os.name == "nt":
    _default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_default):
        _tcmd = _default
if _tcmd:
    pytesseract.pytesseract.tesseract_cmd = _tcmd

# Matches headings like "MP-ECLSS-014.3" or "2.4.1" or "Procedure 7-2"
_HEADING_RE = re.compile(
    r"^\s*((?:MP|FR|AR)-[A-Z]+-\d+(?:\.\d+)?|\d+(?:\.\d+){1,3}|Procedure\s+\d+-\d+)\b", re.MULTILINE
)

OCR_RESOLUTION = 300  # dpi; Tesseract accuracy drops off noticeably below ~250-300
OCR_CONFIG = "--psm 3"  # automatic page segmentation. Measured better than --psm 6 on the
                        # mixed single/multi-column Apollo layouts (88 vs 82 conf on mission
                        # rules); --psm 6 collapsed rotated/tabular checklist pages to noise.

# Parallel, timeout-guarded OCR. Each page is OCR'd in a worker process with a
# hard per-page timeout: a single page that hangs pdfium's renderer (some Apollo
# scans do — it froze the whole run before) is killed and skipped instead of
# being fatal, and OCR spreads across CPU cores.
OCR_WORKERS = int(os.environ.get("LODESTAR_OCR_WORKERS", "4"))
OCR_PAGE_TIMEOUT = int(os.environ.get("LODESTAR_OCR_PAGE_TIMEOUT", "120"))
_WORKER_PDF: dict = {}


def _ocr_page_by_index(pdf_path, page_idx):
    """Worker-process entry point: OCR one page of a PDF, caching the opened
    document per worker so a large PDF isn't re-parsed for every page."""
    pdf = _WORKER_PDF.get(pdf_path)
    if pdf is None:
        pdf = pdfplumber.open(pdf_path)
        _WORKER_PDF[pdf_path] = pdf
    return _ocr_page(pdf.pages[page_idx])


OCR_CONF_RETRY = 70  # below this mean confidence, suspect a rotated/garbled page and try re-orienting


def _run_ocr(image) -> tuple[str, float]:
    """One OCR pass -> (text, mean_word_confidence in [0, 100])."""
    data = pytesseract.image_to_data(image, config=OCR_CONFIG, output_type=pytesseract.Output.DICT)
    words, confidences = [], []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        if not word:
            continue
        words.append(word)
        conf = float(conf)
        if conf >= 0:  # Tesseract emits -1 for non-text regions, not a real confidence
            confidences.append(conf)
    text = " ".join(words)
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, mean_conf


def _osd_angle(image) -> int:
    """Tesseract OSD's suggested clockwise rotation-to-upright (0/90/180/270)."""
    try:
        osd = pytesseract.image_to_osd(image)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0  # OSD needs enough characters; if it can't decide, assume upright


def _ocr_page(page) -> tuple[str, float]:
    """Render the page (pdfplumber -> pypdfium2, no poppler/ImageMagick) and OCR
    it fresh, ignoring any embedded text layer. Returns (text, mean_word_conf).

    Orientation handling: OCR upright first; only if that scores poorly do we
    trust OSD and re-OCR at its suggested rotation — keeping whichever pass is
    more confident. This corrects genuinely-rotated scans (the Apollo checklists)
    WITHOUT letting OSD rotate an already-upright page into garbage (its failure
    mode on mixed-content pages)."""
    image = page.to_image(resolution=OCR_RESOLUTION).original  # PIL.Image
    text, conf = _run_ocr(image)
    if conf < OCR_CONF_RETRY:
        angle = _osd_angle(image)
        if angle:
            t2, c2 = _run_ocr(image.rotate(-angle, expand=True))
            if c2 > conf:
                text, conf = t2, c2
    return text, conf


def extract_raw_segments(pdf_path: str, doc_type: str) -> list[dict]:
    """Returns [{id, raw_text, doc_type, source_page, ocr_confidence}] — no
    LLM calls here. ocr_confidence is the mean Tesseract word-confidence
    (0-100) over the page(s) a segment was drawn from."""
    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
    page_texts = [""] * n_pages
    page_confs = [0.0] * n_pages
    print(f"[pdf_ingest] OCR-ing {n_pages} pages ({OCR_WORKERS} workers) ...", flush=True)
    ex = ProcessPoolExecutor(max_workers=OCR_WORKERS)
    try:
        futures = [(i, ex.submit(_ocr_page_by_index, pdf_path, i)) for i in range(n_pages)]
        for i, fut in futures:
            try:
                page_texts[i], page_confs[i] = fut.result(timeout=OCR_PAGE_TIMEOUT)
            except Exception as e:
                # hung/failed page: skip it (leave text empty) rather than freeze the run
                print(f"[pdf_ingest] page {i + 1}/{n_pages} OCR skipped ({type(e).__name__})", flush=True)
    finally:
        # kill any workers still stuck on a hung page so shutdown doesn't block
        for _p in list(getattr(ex, "_processes", {}).values()):
            try:
                if _p.is_alive():
                    _p.terminate()
            except Exception:
                pass
        ex.shutdown(wait=False)
    print(f"[pdf_ingest] OCR complete: {sum(1 for t in page_texts if t.strip())}/{n_pages} pages with text", flush=True)

    # Track each page's [offset, offset+len) span in the concatenated text so
    # a heading-delimited segment (which frequently crosses a page boundary)
    # can still be attributed a confidence score.
    offsets = []
    pos = 0
    for text in page_texts:
        offsets.append(pos)
        pos += len(text) + 1  # +1 for the "\n" joiner below

    def confidence_for_span(start: int, end: int) -> float:
        confs = [c for off, t, c in zip(offsets, page_texts, page_confs)
                  if off < end and off + len(t) > start]
        return sum(confs) / len(confs) if confs else 0.0

    full_text = "\n".join(page_texts)
    matches = list(_HEADING_RE.finditer(full_text))

    # Heading-based chunking is only trustworthy if headings are DENSE enough.
    # OCR flattens line structure, so on big scanned manuals the regex can fire
    # only 2-3 times across 800+ pages, collapsing the doc into giant chunks the
    # LLM can't extract from (this is exactly what gutted the 844-page AOH:
    # 2 chunks, 0 metadata). Require ~1 heading per 10 pages (min 5), else fall
    # back to per-page segments like every other scanned doc.
    if matches and (len(matches) < 5 or len(matches) * 10 < n_pages):
        print(f"[pdf_ingest] only {len(matches)} headings over {n_pages} pages — "
              f"too sparse, using per-page segments", flush=True)
        matches = []

    if not matches:
        return [{"id": f"{doc_type}-p{i + 1}", "raw_text": t, "doc_type": doc_type,
                  "source_page": i + 1, "ocr_confidence": round(c, 1)}
                for i, (t, c) in enumerate(zip(page_texts, page_confs)) if t.strip()]

    segments = []
    for j, m in enumerate(matches):
        start = m.start()
        end = matches[j + 1].start() if j + 1 < len(matches) else len(full_text)
        seg_id = m.group(1).strip()
        raw = full_text[start:end].strip()
        if len(raw) > 20:  # skip stray heading-only fragments
            segments.append({
                "id": seg_id, "raw_text": raw, "doc_type": doc_type, "source_page": None,
                "ocr_confidence": round(confidence_for_span(start, end), 1),
            })
    return segments
