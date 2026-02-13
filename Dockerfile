FROM python:3.11-slim

WORKDIR /app

# System deps for cartopy/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev libproj-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "climate_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
