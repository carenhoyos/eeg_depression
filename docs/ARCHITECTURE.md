# Arquitectura de la aplicación — apoyo al tamizaje de depresión a partir de EEG

Objetivo: un psiquiatra sube un registro de EEG (archivo de características con la estructura de
entrenamiento, o señal cruda EDF/BDF/CSV), y recibe un informe estandarizado con probabilidad,
clasificación por niveles, gráficos y la ficha de validación del modelo. **Apoyo a la decisión, no diagnóstico.**

## 1. Principios

1. **API primero.** Toda la lógica vive en el backend; el cliente web y el futuro cliente Flutter consumen el
   mismo contrato JSON (`app/backend/schemas.py`, OpenAPI en `/docs`).
2. **Nada fijo en el código.** Niveles de clasificación, prevalencia pre-test, formatos aceptados, colores,
   textos del informe y rutas se leen de `app/config/app.yaml`; el pipeline científico de `config/config.yaml`.
3. **Modelo versionado e inmutable.** `scripts/06_export_bundle.py` congela modelo + configuración + distribuciones
   de referencia + métricas de validación en `models/release/<versión>/bundle.pkl`. El informe cita siempre la versión.
4. **Trazabilidad clínica.** Cada análisis guarda entrada, log de procesamiento, informe JSON, figuras y PDF.
5. **Datos de salud.** Solo pseudónimos en la aplicación; cifrado en reposo y en tránsito; retención configurable;
   auditoría de accesos (fase 2).

## 2. Componentes

```
┌───────────────┐   HTTPS/JSON   ┌───────────────────────────┐        ┌──────────────────────┐
│ Cliente web   │ ─────────────▶ │ API FastAPI               │ ─────▶ │ Worker de análisis   │
│ (HTML/JS hoy, │ ◀───────────── │ /api/v1/analyses          │ cola   │ (hoy BackgroundTasks │
│ Flutter después)│  informe JSON │ /api/v1/model, /config    │        │  mañana Celery/RQ)   │
└───────────────┘                └───────────┬───────────────┘        └──────────┬───────────┘
                                             │                                   │
                                   ┌─────────▼───────────┐              ┌────────▼─────────┐
                                   │ Almacenamiento      │              │ Bundle de modelo │
                                   │ (disco → S3/GCS)    │              │ models/release/  │
                                   │ + metadatos (→ DB)  │              └──────────────────┘
                                   └─────────────────────┘
```

| Capa | Hoy (MVP, este repo) | Escalado (fase siguiente) |
|---|---|---|
| Cliente | `app/frontend`: React 19 + TypeScript + Vite, TanStack Query (estado de servidor), React Router, Recharts/SVG (gráficos desde el JSON) | Flutter (web + iOS + Android) consumiendo la misma API; el cliente React sigue como panel web |
| API | FastAPI + Uvicorn, un proceso | Varias réplicas tras un balanceador; autenticación OIDC; límites de tasa |
| Ejecución | `InProcessQueue` (hilo) tras el puerto `JobQueue` | Adaptador RQ/Celery sobre Redis con workers independientes (misma interfaz) |
| Almacenamiento | `LocalStorage` tras el puerto `Storage` (`app/storage/<id>/`) | Adaptador S3/GCS cifrado + Postgres para metadatos y auditoría |
| Modelo | `bundle.pkl` local | Registro de modelos (MLflow) con promoción por versión y rollback |
| Despliegue | `uvicorn` local o `docker compose up` (Dockerfile multi-etapa: build del frontend + API) | Kubernetes/Cloud Run, IaC, observabilidad (logs estructurados, métricas, trazas) |

### Backend en capas (`app/backend`)

```
core/        config.py (Settings tipadas desde app.yaml + EEGDEP_* env) · dependencies.py (contenedor DI, lifespan)
domain/      schemas.py (contrato Pydantic) · analysis.py (caso de uso: archivo -> informe, sin HTTP)
services/    report.py (niveles, hallazgos, figuras, PDF)
infra/       storage.py (puerto Storage + LocalStorage) · queue.py (puerto JobQueue + InProcessQueue)
api/v1/      routers/meta.py (health, model, config/report) · routers/analyses.py (upload, status, report, figures, pdf)
main.py      create_app(settings): routers, CORS, estáticos del frontend con fallback SPA
```

El caso de uso `run_analysis` no conoce FastAPI: puede ejecutarse igual en un worker remoto.

### Frontend (`app/frontend/src`)

```
api/client.ts          cliente axios tipado (prefijo desde VITE_API_PREFIX)
types.ts               espejo del contrato Pydantic
components/UploadForm  formulario + sondeo de estado (useMutation)
components/ReportView  página del informe (useQuery) con niveles, tablas y figuras
components/charts      ProbabilityBar, BandPowerChart, ContributionsChart, TopographyGrid, CoherenceHeatmap (desde JSON)
App.tsx                rutas: / (nuevo análisis), /history, /analyses/:id
```

## 3. Flujo de un análisis

1. `POST /api/v1/analyses` (multipart): archivo + prevalencia pre-test + metadatos opcionales del registro.
   Responde `202` con `analysis_id`.
2. Worker: detección de formato → (a) CSV de características: `features_from_csv`; (b) señal cruda:
   `read_raw_file` → filtros, notch, referencia, ICA, épocas, rechazo → potencia por banda y coherencia →
   `build_feature_table`. Toda esta ruta usa `config/config.yaml`.
3. `predict_record`: probabilidad calibrada → razón de verosimilitud → **probabilidad posterior a la prevalencia
   pre-test indicada** → nivel (`report.tiers`) → z-scores frente al grupo de referencia → contribuciones (modelo lineal).
4. Figuras (matplotlib, un tono para magnitud, divergente para polaridad), PDF (fpdf2) e informe JSON.
5. `GET /api/v1/analyses/{id}/status` → `GET /api/v1/analyses/{id}` → figuras y `report.pdf`.

## 4. Estructura del informe (estándar)

Combina la estructura de un informe de EEG clínico (identificación, técnica, descripción, impresión,
correlación clínica) con los ítems de TRIPOD+AI que deben acompañar a cualquier salida de un modelo:

1. Identificación y datos del registro (pseudónimo, edad, sexo, condición, medicación, tipo de entrada).
2. Calidad y procesamiento (advertencias, épocas rechazadas, componentes ICA eliminados).
3. Resultado del modelo: probabilidad calibrada, LR, prevalencia pre-test, probabilidad posterior,
   **clasificación por nivel con su texto de impresión**, gráfico de probabilidad sobre los niveles.
4. Descripción cuantitativa: potencia relativa por banda frente a referencia, topomapas z, coherencia,
   desviaciones destacadas y contribuciones del modelo.
5. Modelo y validación (modelo, versión, n de entrenamiento, AUC/sensibilidad/especificidad en CV por sujeto,
   holdout con IC95, calibración).
6. Limitaciones. 7. Aviso legal.

Los niveles por defecto (`app.yaml`): baja probabilidad < 25 %, indeterminado 25–55 %, elevada ≥ 55 %, sobre la
probabilidad posterior. Se eligieron como punto de partida y deben fijarse con el equipo clínico tras un estudio de
utilidad (curvas de decisión), no por conveniencia.

## 5. Cliente Flutter (fase siguiente)

- Un solo proyecto Flutter con targets web, iOS y Android; paquetes `dio` (HTTP), `file_picker`, `fl_chart`
  (gráficos interactivos a partir del JSON: `band_relative_power`, `band_reference`, `findings`, `contributions`).
- Los PNG de la API sirven como respaldo y para el PDF; los gráficos interactivos se construyen del JSON.
- Autenticación con el proveedor de identidad de la institución; nunca se almacenan datos de pacientes en el dispositivo.

## 6. Seguridad, privacidad y regulación (antes de cualquier uso con pacientes)

- Clasificación como software como dispositivo médico (SaMD) en la jurisdicción de uso: FDA (EE. UU.),
  MDR/IVDR (UE), INVIMA (Colombia) u otra; el uso previsto (tamizaje de apoyo) define la clase de riesgo.
- Protección de datos de salud: HIPAA, GDPR o ley local; cifrado, mínimo privilegio, auditoría, retención y borrado.
- Validación clínica prospectiva y multicéntrica antes de reclamar rendimiento; monitoreo de deriva tras el despliegue.
- Licencia y consentimiento de la base de entrenamiento: confirmar que permiten uso en un producto.

## 7. Ejecución local

```bash
python scripts/06_export_bundle.py            # tras 04_train_eval.py: crea models/release/<versión>/bundle.pkl
(cd app/frontend && npm install && npm run build)
python -m uvicorn app.backend.main:app --port 8000
python -m pytest -q tests
# abrir http://localhost:8000  (docs de la API en /api/v1/docs) · docker compose up --build
```
