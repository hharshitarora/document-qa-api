# Slim rather than alpine: the wheels this project needs are built for glibc, and
# alpine would trade image size for compiling them.
FROM python:3.11-slim

# Keep .pyc files out of the image and logs unbuffered so they reach the platform in
# real time rather than sitting in a buffer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies in their own layer, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY static/ static/
COPY scripts/ scripts/
COPY samples/ samples/

# Run as a non-root user: nothing here needs root, and a container that does not need
# it should not have it.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# The key is passed in at run time and never baked into the image.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
