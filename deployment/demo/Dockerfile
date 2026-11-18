FROM python:3.13-slim AS python-base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

FROM python-base AS native-build
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake g++ make \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir pybind11
COPY cpp cpp
RUN cmake -S cpp -B /tmp/build -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PYTHON_BINDINGS=ON -DPython_EXECUTABLE="$(command -v python)" \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build /tmp/build --target retrieval_cpp --parallel 2 \
    && python cpp/tools/convert_corpus.py cpp/tests/fixtures/demo.csv /app/demo.bin

FROM python-base AS api
COPY backend/__init__.py backend/
COPY backend/api backend/api
COPY --from=native-build /app/backend/retrieval_cpp*.so backend/
COPY --from=native-build /app/demo.bin /app/demo.bin
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/api"]

FROM node:24-bookworm-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/public public
COPY frontend/src src
RUN VITE_API_BASE=/api npm run build

FROM caddy:2.10-alpine AS web
COPY deployment/demo/Caddyfile /etc/caddy/Caddyfile
COPY --from=frontend-build /frontend/dist /srv
