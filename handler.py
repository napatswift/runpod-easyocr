import io
import os
from typing import Any, Dict, List, Optional, Tuple

import runpod
import requests

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "PyMuPDF (pymupdf) is required. Ensure it's installed."
    ) from e

try:
    import easyocr
except Exception as e:
    raise RuntimeError("easyocr is required. Ensure it's installed.") from e

try:
    import numpy as np
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "Pillow and numpy are required for batched processing."
    ) from e


# Global cache for EasyOCR reader to avoid reloading weights on every request.
_READER_CACHE: Dict[Tuple[Tuple[str, ...], bool, bool], easyocr.Reader] = {}


def get_reader(
    languages: List[str], use_gpu: bool, cudnn_benchmark: bool = False
) -> easyocr.Reader:
    key = (tuple(languages), bool(use_gpu), bool(cudnn_benchmark))
    if key not in _READER_CACHE:
        _READER_CACHE[key] = easyocr.Reader(
            languages, gpu=use_gpu, cudnn_benchmark=cudnn_benchmark
        )
    return _READER_CACHE[key]


def download_pdf(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def pdf_pages_to_png_bytes(
    pdf_bytes: bytes,
    dpi: int = 200,
    page_indices: Optional[List[int]] = None,
    page_from: Optional[int] = None,
    page_to: Optional[int] = None,
    page_limit: Optional[int] = None,
) -> List[Tuple[int, bytes]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    total_pages = len(doc)
    selected: List[int] = []

    if page_indices is not None:
        selected = [p for p in page_indices if 0 <= p < total_pages]
    else:
        start = page_from if page_from is not None else 0
        end = page_to if page_to is not None else total_pages - 1
        start = max(0, start)
        end = min(total_pages - 1, end)
        selected = list(range(start, end + 1))

    if page_limit is not None:
        selected = selected[: max(0, int(page_limit))]

    # Render pages at requested DPI
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: List[Tuple[int, bytes]] = []
    for page_index in selected:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        images.append((page_index, img_bytes))
    return images


def image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes (PNG/JPEG) to RGB numpy array without OpenCV."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        return np.array(im)


def ocr_image_bytes(
    reader: easyocr.Reader, image_bytes: bytes, detail: int = 1
):
    # easyocr supports file path, numpy array, or bytes; we pass bytes.
    return reader.readtext(image_bytes, detail=detail)


def normalize_results(results: List[Any], detail: int) -> List[Dict[str, Any]]:
    if detail == 0:
        simplified = []
        for t in results:
            # If the library returned detailed tuples, extract text; else pass as-is
            if (
                isinstance(t, (list, tuple))
                and len(t) >= 2
                and isinstance(t[1], str)
            ):
                simplified.append({"text": t[1]})
            else:
                simplified.append({"text": t if isinstance(t, str) else str(t)})
        return simplified
    normalized = []
    for item in results:
        # item = [bbox, text, confidence]
        try:
            bbox, text, conf = item
        except Exception:
            # If format is unexpected, pass-through
            normalized.append({"raw": item})
            continue
        normalized.append(
            {
                "box": bbox,
                "text": text,
                "confidence": float(conf),
            }
        )
    return normalized


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runpod serverless handler.

    Expected input JSON:
    {
      "pdf_urls": ["https://...", ...] | "pdf_url": "https://...",
      "languages": ["ch_sim", "en"],
      "gpu": true,
      "detail": 1,
      "dpi": 200,
      "page_indices": [0,1],
      "page_from": 0,
      "page_to": 5,
      "page_limit": 10
    }
    """
    inp = event.get("input") or {}

    # Accept single url via "pdf_url" or multiple via "pdf_urls"
    pdf_urls = inp.get("pdf_urls")
    if isinstance(pdf_urls, str):
        pdf_urls = [pdf_urls]
    if not pdf_urls:
        single = inp.get("pdf_url")
        pdf_urls = [single] if single else []

    if not pdf_urls:
        return {"error": "Missing 'pdf_url' or 'pdf_urls' in input."}

    # Defaults
    default_langs = os.getenv("READER_LANGS", "ch_sim,en").split(",")
    languages: List[str] = inp.get("languages") or [
        l.strip() for l in default_langs if l.strip()
    ]
    use_gpu: bool = bool(inp.get("gpu", True))
    detail: int = int(inp.get("detail", 1))
    dpi: int = int(inp.get("dpi", 200))
    page_indices = inp.get("page_indices")
    page_from = inp.get("page_from")
    page_to = inp.get("page_to")
    page_limit = inp.get("page_limit")
    batched: bool = bool(inp.get("batched", False))
    n_width: Optional[int] = inp.get("n_width")
    n_height: Optional[int] = inp.get("n_height")
    cudnn_benchmark: bool = bool(inp.get("cudnn_benchmark", False))

    # Initialize reader once per request based on languages + gpu
    reader = get_reader(languages, use_gpu, cudnn_benchmark=cudnn_benchmark)

    outputs = []
    for url in pdf_urls:
        item: Dict[str, Any] = {"url": url, "pages": []}
        try:
            pdf_bytes = download_pdf(url)
            page_images = pdf_pages_to_png_bytes(
                pdf_bytes,
                dpi=dpi,
                page_indices=page_indices,
                page_from=page_from,
                page_to=page_to,
                page_limit=page_limit,
            )

            if not batched:
                for idx, img_bytes in page_images:
                    results = ocr_image_bytes(reader, img_bytes, detail=detail)
                    item["pages"].append(
                        {
                            "index": idx,
                            "results": normalize_results(
                                results, detail=detail
                            ),
                        }
                    )
            else:
                # Prepare batch as numpy arrays
                arrays: List[np.ndarray] = []
                indices: List[int] = []
                widths = []
                heights = []
                for idx, img_bytes in page_images:
                    arr = image_bytes_to_array(img_bytes)
                    h, w = arr.shape[:2]
                    heights.append(h)
                    widths.append(w)
                    arrays.append(arr)
                    indices.append(idx)

                # If sizes differ and no target provided, choose a common size
                nw, nh = n_width, n_height
                if (nw is None or nh is None) and (
                    len(set(widths)) > 1 or len(set(heights)) > 1
                ):
                    # Pick max dims as a simple heuristic
                    nw = max(widths)
                    nh = max(heights)

                # Run batched OCR. Avoid passing detail explicitly for compatibility.
                # EasyOCR returns a list per image.
                batched_results = reader.readtext_batched(
                    arrays, n_width=nw, n_height=nh
                )

                # Map back to pages
                for idx, res in zip(indices, batched_results):
                    item["pages"].append(
                        {
                            "index": idx,
                            "results": normalize_results(res, detail=detail),
                        }
                    )

                # Keep output pages sorted by index
                item["pages"].sort(key=lambda p: p["index"])

        except Exception as e:
            item["error"] = str(e)
        outputs.append(item)

    return {
        "results": outputs,
        "languages": languages,
        "gpu": use_gpu,
        "detail": detail,
        "dpi": dpi,
        "batched": batched,
        "n_width": n_width,
        "n_height": n_height,
        "cudnn_benchmark": cudnn_benchmark,
    }


runpod.serverless.start({"handler": handler})
