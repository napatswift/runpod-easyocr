FROM runpod/pytorch:3.10-2.1.2-12.1.0

WORKDIR /app

# Optional system packages sometimes needed by OpenCV and fonts used by EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    READER_LANGS=ch_sim,en

CMD ["python", "-u", "handler.py"]

