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
└── requirements.txt
```

## Uso

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_inspect.py --config config/config.yaml
.venv/bin/python scripts/02_standardize_qc.py
.venv/bin/python scripts/03_features.py
.venv/bin/python scripts/04_train_eval.py
.venv/bin/python scripts/05_interpret.py
# o: scripts/run_all.sh config/config.yaml
```

Para experimentar, copia el YAML (p. ej. `config/exp_coh_all.yaml`) y cambia solo lo necesario
(`features.coherence.mode: all`, `validation.outer.scheme: loso`, etc.).

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
