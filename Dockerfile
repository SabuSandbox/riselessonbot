FROM python:3.11-slim

# Install tesseract if you want OCR (optional)
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .
ENV PORT=5000
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app --workers 1
