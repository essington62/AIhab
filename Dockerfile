FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY environment_docker.txt .
RUN pip install --no-cache-dir -r environment_docker.txt

# Copy project
COPY src/ src/
COPY conf/ conf/
COPY scripts/ scripts/
COPY CLAUDE.md .

# Create data directories
RUN mkdir -p data/{01_raw/{spot,futures,macro,coinglass,sentiment,news,market},02_intermediate/{spot,futures,macro},02_features,03_models,04_scoring,05_output} \
    && mkdir -p logs

# Make scripts executable
RUN chmod +x scripts/*.sh

# Crontab setup
COPY crontab /etc/cron.d/aihab-cron
RUN chmod 0644 /etc/cron.d/aihab-cron \
    && crontab /etc/cron.d/aihab-cron

# Default: run cron in foreground
CMD ["cron", "-f"]
