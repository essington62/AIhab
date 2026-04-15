FROM python:3.11-slim
WORKDIR /app

# System dependencies + tini (PID 1) + supercronic (cron)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      tini \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://github.com/aptible/supercronic/releases/download/v0.2.33/supercronic-linux-amd64 \
       -o /usr/local/bin/supercronic \
    && chmod +x /usr/local/bin/supercronic

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

# Crontab
COPY crontab /etc/cron.d/aihab-cron

# PYTHONPATH baked in — visible to supercronic jobs and any python -m call
ENV PYTHONPATH=/app

# tini as PID 1 (proper signal handling + zombie reaping)
# supercronic: runs in foreground, inherits env, logs to stdout
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/supercronic", "/etc/cron.d/aihab-cron"]