FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for Matplotlib & LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--chdir", "src", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
