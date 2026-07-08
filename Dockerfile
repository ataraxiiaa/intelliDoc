FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (including poppler-utils for pdf2image PDF rendering)
RUN apt-get update && apt-get install -y \
    libpq-dev gcc poppler-utils libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Copy code modules
COPY src/ src/
COPY sample_data/ sample_data/

# Default command to run FastAPI web app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
