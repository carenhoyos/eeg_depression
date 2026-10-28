# eeg_depression — tamizaje de depresión a partir de características de EEG (validación por sujeto)

Pipeline modular y reproducible. **Todo parámetro vive en `config/config.yaml`**; las funciones reciben
los parámetros como argumentos y los scripts solo leen la configuración.

## Estructura

```
eeg_depression/
├── config/config.yaml          # única fuente de parámetros (rutas, columnas, canales, bandas, QC, modelos, validación)
├── data/raw/                   # CSV original (no se modifica)
├── src/eegdep/
│   ├── config.py               # carga/validación del YAML, semillas, rutas
│   ├── io.py                   # CSV -> EEGFeatureSet (n, banda, canal) y (n, banda, canal, canal)
│   ├── qc.py                   # duplicados, z robusto, rango de coherencia, política de descarte
│   ├── preprocess.py           # filtros/notch/rereferencia/ICA/épocas con MNE (solo data.kind == raw)
│   ├── features.py             # potencia relativa, regional, asimetría, coherencia regional/completa
│   ├── models.py               # pipelines sklearn (escalado -> selección -> clasificador) desde YAML
│   ├── validation.py           # holdout por sujeto, CV anidada agrupada, bootstrap, calibración, permutación
│   └── interpret.py            # efectos univariados, importancias, topomapas (MNE), calibración
├── scripts/
│   ├── 01_inspect.py           # inspección guiada por configuración
│   ├── 02_standardize_qc.py    # estandarización + QC + reporte de descartes
│   ├── 03_features.py          # tabla de características por sujeto
│   ├── 04_train_eval.py        # entrenamiento/validación de todos los modelos
│   ├── 05_interpret.py         # interpretabilidad y calibración
│   └── run_all.sh              # ejecuta 01..05
├── outputs/                    # artefactos numerados por etapa (json/csv/pkl/figures)
├── reports/TRIPOD_AI_template.md
├── models/release/<versión>/   # bundle versionado del modelo (06_export_bundle.py); `latest` apunta a la versión activa
├── app/
│   ├── config/app.yaml         # configuración de la aplicación (niveles, prevalencia, formatos, textos, estilo)
│   ├── backend/                # FastAPI en capas: core (config, DI) · domain (contratos, caso de uso) · services (informe) · infra (storage, cola) · api/v1
│   └── frontend/               # React + TypeScript + Vite + TanStack Query + Recharts (build en dist/)
├── tests/test_api.py           # pruebas de integración de la API
├── docs/ARCHITECTURE.md        # arquitectura, flujo, estándar del informe, plan Flutter y regulación
├── Dockerfile · docker-compose.yml
└── requirements.txt · requirements-app.txt
```

## Uso

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/01_inspect.py --config config/config.yaml
python scripts/02_standardize_qc.py
python scripts/03_features.py
python scripts/04_train_eval.py
python scripts/05_interpret.py
# o: scripts/run_all.sh config/config.yaml   (PY=/otro/python para cambiar el intérprete)
```

Para experimentar, copia el YAML (p. ej. `config/exp_coh_all.yaml`) y cambia solo lo necesario
(`features.coherence.mode: all`, `validation.outer.scheme: loso`, etc.).

## Segundo conjunto de datos: señal cruda (OpenNeuro ds003478)

`config/config_ds003478.yaml` activa el pipeline de señal cruda sobre el dataset BIDS "EEG: Depression rest"
(Cavanagh et al., CC0; 122 universitarios, 64 canales, 500 Hz, ojos abiertos/cerrados). Etiqueta: BDI ≥ 13 frente a
BDI < 7 (zona intermedia excluida), exclusiones del README original, bloques de ojos cerrados de la corrida 1.
Por sujeto: remuestreo, notch, pasabanda, referencia promedio sobre 64 canales, ICA (cacheada), selección de los 19
canales 10-20, épocas de 4 s con rechazo, potencia por banda y coherencia. Los datos viven fuera del repo (`data.bids_root`).

```bash
python -m openneuro download --dataset ds003478 --tag 1.1.0 --target_dir ../data/ds003478 --include "sub-*/eeg/*run-01*" --include participants.tsv
for s in 02_standardize_qc 03_features 04_train_eval 05_interpret 07_deep_eegnet 08_cross_site; do python scripts/$s.py --config config/config_ds003478.yaml; done
python scripts/06_export_bundle.py --config config/config_ds003478.yaml --version 0.2.0
```

`07_deep_eegnet.py` entrena EEGNet (PyTorch) sobre las épocas crudas con el mismo holdout y validación por sujeto;
`08_cross_site.py` aplica el modelo final a las características del otro conjunto (transferencia entre sitios).

## Aplicación web (API + cliente React)

```bash
pip install -r requirements-app.txt
python scripts/06_export_bundle.py                      # publica models/release/<versión>/bundle.pkl
(cd app/frontend && cp .env.example .env && npm install && npm run build)
python -m uvicorn app.backend.main:app --port 8000      # http://localhost:8000 · API docs en /api/v1/docs
python -m pytest -q tests                               # pruebas de integración
```

Desarrollo del frontend con recarga en caliente: `cd app/frontend && npm run dev` (reenvía la API al backend
definido en `VITE_DEV_BACKEND`). Con Docker: `docker compose up --build`.

Rol de usuario: psiquiatra (JWT; usuario inicial definido en `auth.*` del `app.yaml`, sobreescribible por entorno).
Pacientes identificados por pseudónimo, con historial de análisis y evolución de la probabilidad.
Entradas aceptadas (configurables en `app/config/app.yaml`): señal cruda EDF/BDF, EEGLAB (.set + .fdt), FIF o CSV
(requiere frecuencia de muestreo), y CSV de características con la estructura de entrenamiento
(`data/examples/sample_features.csv`). Con señal cruda el informe incluye un visor de la señal limpia y su espectro.
El informe (JSON, figuras y PDF) incluye probabilidad calibrada, razón de verosimilitud, probabilidad posterior a la
prevalencia pre-test indicada, nivel de clasificación, desviaciones frente al grupo de referencia, contribuciones del
modelo y la ficha de validación. Ver `docs/ARCHITECTURE.md`.

## Garantías de validación

- Toda partición (holdout, CV externa, CV interna, calibración) usa `StratifiedGroupKFold` con el grupo = sujeto.
- Escalado, selección de características e hiperparámetros se ajustan solo con el entrenamiento de cada partición.
- El holdout de sujetos se evalúa una sola vez por modelo, con el modelo final ajustado en todo el desarrollo.
- IC95 por bootstrap sobre sujetos; prueba de permutación de etiquetas a nivel sujeto.

## Supuestos declarados (ver limitaciones en el reporte TRIPOD+AI)

- El CSV no tiene identificador de sujeto: se asume **1 fila = 1 registro = 1 sujeto** (`data.subject_column: null`).
  Las filas con bloque de potencia idéntico se tratan como el mismo sujeto (`qc.duplicate_policy: same_group`).
- Frecuencia de muestreo, referencia, condición ocular y equipo no se deducen del archivo: `acquisition.*` queda en `null`
  hasta confirmarlo con la fuente.
