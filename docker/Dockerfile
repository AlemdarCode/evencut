# syntax=docker/dockerfile:1
# Evencut: HTTP service + command-line tool in one small image.
#
#   docker build -f docker/Dockerfile -t evencut .
#   docker run --rm -p 127.0.0.1:8000:8000 evencut                 # service on localhost only
#   echo "5 3 8 2 7 4 6 1 9 5" | docker run --rm -i evencut evencut -k 3   # CLI
#
# The service has no authentication: publish it on localhost or behind your own gateway.

# ---- build stage: compile the native library (inside the wheel) and the CLI ----------------
FROM python:3.13-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt pyproject.toml README.md LICENSE-MIT LICENSE-APACHE ./
COPY include include
COPY src src
COPY cli cli
COPY python python
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels . \
 && cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DEVENCUT_BUILD_TESTS=OFF -DEVENCUT_BUILD_SHARED=OFF -DEVENCUT_BUILD_CLI=ON \
 && cmake --build /build --target evencut_cli

# ---- runtime stage --------------------------------------------------------------------------
FROM python:3.13-slim
RUN useradd --create-home --uid 10001 evencut
COPY --from=build /wheels /wheels
COPY --from=build /build/bin/evencut /usr/local/bin/evencut
# versions tested with this release
RUN pip install --no-cache-dir /wheels/*.whl fastapi==0.162.0 uvicorn==0.54.0 \
 && rm -rf /wheels
USER evencut
ENV EVENCUT_MAX_ITEMS=2000000 \
    EVENCUT_MAX_BODY_BYTES=67108864 \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"
CMD ["uvicorn", "evencut.service:app", "--host", "0.0.0.0", "--port", "8000"]
