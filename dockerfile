# ===== Base Image =====
FROM python:3.13-slim

# ===== Set working directory =====
WORKDIR /app

# ===== System Dependencies for Selenium & Chrome =====
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    fonts-liberation \
    libnss3 \
    libxss1 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libx11-xcb1 \
    libgtk-3-0 \
    curl \
    wget \
    unzip \
    git \
    build-essential \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Set Chromium environment variables
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_PATH=/usr/lib/chromium/

# ===== Copy requirements & install dependencies first (Docker caching) =====
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# ===== Copy application code =====
COPY . .

# ===== Expose port for FastAPI =====
EXPOSE 8000

# ===== Set entrypoint =====
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
