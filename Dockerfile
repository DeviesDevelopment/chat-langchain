FROM python:3.11-bookworm

# Install system packages OpenCV needs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.5.1

RUN poetry config virtualenvs.create false

# Create a non-root user (UID 1000 to match ACA)
RUN useradd -u 1000 appuser

COPY ./pyproject.toml ./poetry.lock* ./

COPY ./backend/*.py ./backend/

RUN poetry install --no-interaction --no-ansi

# Create all known write paths (add yours here)
RUN mkdir -p /tmp/chroma_db /tmp/nltk_data && \
    chown -R 1000:1000 /tmp/chroma_db /tmp/nltk_data

# Pre-download NLTK corpora if needed (example: punkt)
RUN python -m nltk.downloader -d /tmp/nltk_data punkt punkt_tab

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER 1000

# Run entrypoint script when container starts
CMD ["./entrypoint.sh"]

