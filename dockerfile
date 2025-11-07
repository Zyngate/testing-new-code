# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget curl unzip xvfb libnss3 libgconf-2-4 libxi6 libxrender1 libglib2.0-0 \
    chromium chromium-driver git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Chrome/Selenium
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_DRIVER=/usr/bin/chromedriver

# Set workdir
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Expose port (Railway default for web service)
EXPOSE 8080

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
