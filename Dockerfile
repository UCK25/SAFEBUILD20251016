FROM python:3.10-slim

WORKDIR /app

# system deps for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_web.txt ./
RUN pip install --no-cache-dir -r requirements_web.txt

COPY . /app

EXPOSE 8000
CMD ["uvicorn", "web_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
