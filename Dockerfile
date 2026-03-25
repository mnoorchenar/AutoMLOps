FROM python:3.11-slim

# Install system dependencies required by LightGBM (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

RUN mkdir -p mlruns logs

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300", "--log-level", "info", "app:app"]
