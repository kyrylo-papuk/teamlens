FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# No `ml` extra: the image stays small. Triage then uses Claude or the word list, duplicate search uses words.
RUN pip install .

RUN useradd --create-home app && mkdir /app/data && chown app /app/data
USER app

# Inside a container the services must listen on all interfaces; the compose file decides what is published.
ENV TEAMLENS_HOST=0.0.0.0
EXPOSE 8000 9101
CMD ["python", "-m", "teamlens.web"]
