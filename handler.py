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

try:
    import pytesseract
    from pytesseract import Output as TessOutput
except Exception as e:
    raise RuntimeError(
        "pytesseract is required for orientation detection."
    ) from e


def _to_py_scalar(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return _to_py_scalar(obj)


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
) -> List[Tuple[int, bytes, Tuple[int, int]]]:
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

    images: List[Tuple[int, bytes, Tuple[int, int]]] = []
    for page_index in selected:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        images.append((page_index, img_bytes, (pix.width, pix.height)))
    return images


def image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes (PNG/JPEG) to RGB numpy array without OpenCV."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        return np.array(im)


def image_bytes_size(image_bytes: bytes) -> Tuple[int, int]:
    """Deprecated: prefer sizes returned by pdf_pages_to_png_bytes."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        return im.size


def correct_orientation_bytes(
    image_bytes: bytes,
) -> Tuple[bytes, Tuple[int, int], Dict[str, Any]]:
    """
    Detect orientation via Tesseract OSD and rotate the image bytes if needed.
    Returns (new_bytes, (width,height), osd_dict).
    """
    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb = im.convert("RGB")
        osd: Dict[str, Any] = {}
        rotate_degrees = 0
        try:
            osd = pytesseract.image_to_osd(rgb, output_type=TessOutput.DICT)
            rotate_degrees = int(osd.get("rotate", 0) or 0)
        except Exception:
            # If OSD fails, proceed without rotation
            osd = {"rotate": 0}

        rotated = rgb
        if rotate_degrees % 360 != 0:
            rotated = rgb.rotate(0 - rotate_degrees, expand=True)

        buf = io.BytesIO()
        rotated.save(buf, format="PNG")
        data = buf.getvalue()
        w, h = rotated.size
        return data, (w, h), osd


def ocr_image_bytes(
    reader: easyocr.Reader, image_bytes: bytes, detail: int = 1
):
    # easyocr supports file path, numpy array, or bytes; we pass bytes.
    return reader.readtext(image_bytes, detail=detail)


def normalize_results(
    results: List[Any],
    detail: int,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[Dict[str, Any]]:
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
        # Normalize bbox coordinates if width/height are provided; else just coerce types
        safe_box: List[List[float]] = []
        if width and height and width > 0 and height > 0:
            try:
                for pt in bbox:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        x = float(_to_py_scalar(pt[0])) / float(width)
                        y = float(_to_py_scalar(pt[1])) / float(height)
                        # Clamp to [0.0, 1.0]
                        x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
                        y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
                        safe_box.append([x, y])
            except Exception:
                # Fallback to json-safe conversion if unexpected structure
                safe_box = json_safe(bbox)
        else:
            try:
                for pt in bbox:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        safe_box.append(
                            [
                                float(_to_py_scalar(pt[0])),
                                float(_to_py_scalar(pt[1])),
                            ]
                        )
                    else:
                        val = json_safe(pt)
                        # Ensure numeric types if possible
                        if isinstance(val, (int, float)):
                            safe_box.append([float(val), 0.0])
                        else:
                            safe_box.append(val)
            except Exception:
                safe_box = json_safe(bbox)

        normalized.append(
            {"box": safe_box, "text": text, "confidence": float(conf)}
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
                for idx, img_bytes, (w, h) in page_images:
                    corrected_bytes, (cw, ch), osd = correct_orientation_bytes(
                        img_bytes
                    )
                    results = ocr_image_bytes(
                        reader, corrected_bytes, detail=detail
                    )
                    page_out = {
                        "index": idx,
                        "results": normalize_results(
                            results, detail=detail, width=cw, height=ch
                        ),
                    }
                    # Attach minimal orientation metadata
                    if isinstance(osd, dict):
                        page_out["orientation"] = {
                            "rotate": int(osd.get("rotate", 0) or 0),
                            "script": osd.get("script"),
                        }
                    item["pages"].append(page_out)
            else:
                # Prepare batch as numpy arrays
                arrays: List[np.ndarray] = []
                indices: List[int] = []
                widths = []
                heights = []
                orientations: List[Dict[str, Any]] = []
                for idx, img_bytes, (w, h) in page_images:
                    corrected_bytes, (cw, ch), osd = correct_orientation_bytes(
                        img_bytes
                    )
                    arr = image_bytes_to_array(corrected_bytes)
                    arrays.append(arr)
                    indices.append(idx)
                    widths.append(cw)
                    heights.append(ch)
                    orientations.append(
                        {
                            "rotate": int(osd.get("rotate", 0) or 0),
                            "script": osd.get("script"),
                        }
                    )

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
                for idx, res, w, h, orient in zip(
                    indices, batched_results, widths, heights, orientations
                ):
                    eff_w = nw if nw else w
                    eff_h = nh if nh else h
                    item["pages"].append(
                        {
                            "index": idx,
                            "results": normalize_results(
                                res, detail=detail, width=eff_w, height=eff_h
                            ),
                            "orientation": orient,
                        }
                    )

                # Keep output pages sorted by index
                item["pages"].sort(key=lambda p: p["index"])

        except Exception as e:
            item["error"] = str(e)
        outputs.append(item)

    return json_safe(
        {
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
    )


runpod.serverless.start({"handler": handler})
