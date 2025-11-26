FROM python:3.10-slim

# Install OS dependencies required for OpenCV & face processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render uses port 10000
EXPOSE 10000

# Run app with gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
