# docs/TECH_STACK.md "Deployment: Docker + Docker Compose" — Python base image, installs requirements.txt,
# runs streamlit. Built from the repo root (see docker-compose.yml's build.context).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
