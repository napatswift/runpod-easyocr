# Runpod EasyOCR (PDF)

[![Runpod](https://api.runpod.io/badge/napatswift/runpod-easyocr)](https://console.runpod.io/hub/napatswift/runpod-easyocr)

Serverless OCR for PDF files using EasyOCR on Runpod. Provide public PDF URLs and get extracted text with bounding boxes (normalized to 0..1) and confidence per page.

## Input schema

- `pdf_url` or `pdf_urls`: Single URL string or list of public PDF URLs.
- `languages`: List of language codes for EasyOCR (default `['ch_sim','en']`).
- `gpu`: Boolean to use GPU if available (default `true`).
- `detail`: `1` for boxes, text, confidence; `0` for text only (default `1`).
- `dpi`: Rendering DPI for PDF to image (default `200`).
- `page_indices`: List of zero-based page indices to process.
- `page_from`/`page_to`: Page range (inclusive) to process.
- `page_limit`: Max number of pages to process.
- `batched`: Use EasyOCR `readtext_batched` to process pages in a batch (default `false`).
- `n_width`/`n_height`: When batching, resize all pages to these exact dimensions. If not provided and pages differ in size, the largest width/height are used.
- `cudnn_benchmark`: Enables cuDNN benchmark mode for consistent batch sizes (default `false`).

## Example request body

```json
{
  "pdf_urls": [
    "https://arxiv.org/pdf/1708.01204.pdf"
  ],
  "languages": ["ch_sim", "en"],
  "gpu": true,
  "detail": 1,
  "dpi": 200,
  "page_limit": 1,
  "batched": true,
  "n_width": 1200,
  "n_height": 1600,
  "cudnn_benchmark": true
}
```

## Output shape

```json
{
  "results": [
    {
      "url": "https://...",
      "pages": [
        {
          "index": 0,
          "results": [
            { "box": [[x,y],...], "text": "...", "confidence": 0.99 }
          ]
        }
      ]
    }
  ],
  "languages": ["ch_sim","en"],
  "gpu": true,
  "detail": 1,
  "dpi": 200
}
```

## Local testing

You can run the handler locally by setting `INPUT_JSON` and executing the file, or by using the Runpod testing CLI. This repo also includes `.runpod/tests.json` for Hub automated tests.

## Deployment

1. Ensure Docker is available and build the image: `docker build -t runpod-easyocr .`
2. Push to your registry or connect the repo to Runpod Hub.
3. Create a release on GitHub to trigger Hub ingestion.

## Notes

- The worker uses PyMuPDF to render PDF pages to images, avoiding external system dependencies.
- The EasyOCR Reader is cached between requests to avoid reloading weights.
- Set default languages via env `READER_LANGS` (e.g., `ch_sim,en`).
 - Output `box` coordinates are normalized: `x` in [0.0,1.0] relative to image width and `y` in [0.0,1.0] relative to image height.
