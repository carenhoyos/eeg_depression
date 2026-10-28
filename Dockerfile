# Imagen de la API (frontend ya compilado se copia en app/frontend/dist)
FROM node:20-alpine AS frontend
WORKDIR /fe
COPY app/frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY app/frontend ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
COPY requirements.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-app.txt
COPY src ./src
COPY config ./config
COPY app/config ./app/config
COPY app/backend ./app/backend
COPY models/release ./models/release
COPY --from=frontend /fe/dist ./app/frontend/dist
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
