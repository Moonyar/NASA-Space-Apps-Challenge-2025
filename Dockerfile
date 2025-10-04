FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8080
ENV GUNICORN_CMD_ARGS="--workers=1 --threads=8 --timeout=120"
CMD ["gunicorn", "-b", ":8080", "app:app"]