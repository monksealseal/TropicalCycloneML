FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_suitability_framework/ ai_suitability_framework/
COPY service/ service/

ENV TERRASCORE_PORT=8080
ENV TERRASCORE_SECRET_KEY=change-me-in-production

EXPOSE 8080

CMD ["python", "-m", "service.app"]
