# Slim image aimed at Render Free.
# Install client build tools first, then switch to production.

FROM node:20-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHON=python3

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy psutil \
    && rm -rf /root/.cache/pip

COPY package.json package-lock.json ./
COPY client/package.json client/package-lock.json ./client/

# Must install client devDeps (vite) for build — do NOT set NODE_ENV=production yet
RUN npm ci \
    && npm ci --prefix client

COPY . .

RUN npm run build --prefix client \
    && rm -rf client/node_modules client/src \
    && npm prune --omit=dev \
    && npm cache clean --force

ENV NODE_ENV=production

EXPOSE 10000
CMD ["npm", "start"]
