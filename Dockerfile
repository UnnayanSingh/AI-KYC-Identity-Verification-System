# --------------------------
# BASE IMAGE (Python 3.10)
# --------------------------
FROM python:3.10-slim

# Avoid prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# --------------------------
# INSTALL SYSTEM DEPENDENCIES
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    wget \
    curl \
    git \
    libopencv-dev \
    python3-opencv \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------------
# CREATE WORKDIR
# --------------------------
WORKDIR /app

# Copy project files
COPY . /app

# --------------------------
# INSTALL PYTHON DEPENDENCIES
# --------------------------
RUN pip install --upgrade pip

# Install face_recognition & dependencies
RUN pip install \
    wheel \
    numpy \
    Pillow \
    cmake \
    dlib \
    face_recognition \
    opencv-python \
    pytesseract \
    reportlab \
    flask \
    deepface || true

# --------------------------
# EXPOSE PORT
# --------------------------
EXPOSE 5000

# --------------------------
# START THE FLASK APP
# --------------------------
CMD ["python", "app.py"]
