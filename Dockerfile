FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir . \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir "opencv-python-headless<5,>=4.11.0.86"

COPY app ./app
COPY frontend ./frontend

EXPOSE 8000

CMD ["python", "-m", "app.server"]
