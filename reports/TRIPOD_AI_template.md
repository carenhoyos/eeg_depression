# Reporte TRIPOD+AI — Modelo de apoyo al tamizaje de depresión mayor a partir de características de EEG en reposo

> Plantilla alineada con TRIPOD+AI (Collins et al., BMJ 2024). Cada ítem indica dónde se obtiene la información en este repositorio (`outputs/`, `config/config.yaml`). Los campos entre corchetes se rellenan al cerrar el estudio. Uso previsto: **tamizaje y apoyo al profesional, no diagnóstico autónomo**.

## Título y resumen
- **1. Título.** Identificar el estudio como desarrollo (y validación interna) de un modelo de predicción, la población (adultos, Corea del Sur, un solo sitio), el desenlace (diagnóstico clínico de trastorno depresivo vs. control sano) y que se usa IA/ML.
- **2. Resumen estructurado.** Objetivo, fuente de datos, participantes (n = 294; 199 depresión / 95 control), predictores (potencia por banda y coherencia, 19 canales 10-20), desenlace, modelos (SVM lineal/RBF, Random Forest, regresión logística), validación (CV anidada agrupada por sujeto + holdout de sujetos), resultados (AUC con IC95 de `outputs/04_results.json`), limitaciones principales (sin ID de sujeto, sin demografía, un sitio).

## Introducción
- **3a. Contexto clínico y justificación.** Carga de la depresión, limitaciones de las escalas de autoreporte, racional neurofisiológico de los marcadores EEG (potencia, asimetría, conectividad) y evidencia previa, incluyendo su heterogeneidad y problemas de replicación.
- **3b. Objetivos.** Desarrollar y validar internamente un modelo de tamizaje; explicitar que NO se valida externamente en este estudio.

## Métodos
### Fuente de datos
- **4a. Diseño y fuente.** [Describir la base original: institución, periodo de reclutamiento, cómo se obtuvo el archivo `machinelearning_data_EEG.csv`, licencia y consentimiento. Pendiente de confirmación por la autora.]
- **4b. Fechas.** [Fechas de registro EEG y de diagnóstico.]
- **5a. Criterios de elegibilidad.** [Criterios de inclusión/exclusión de la base original; criterios de este estudio: filas con etiqueta en {Depressive disorder, Healthy control}.]
- **5b. Tratamientos/medicación.** [Estado de medicación psicotrópica en el momento del registro. Si no está disponible, declararlo como limitación: confusor plausible de potencia beta/alfa.]
- **6a. Desenlace.** Diagnóstico clínico (entrevista estructurada o criterio DSM/ICD, [especificar]). Codificación: `data.positive_class`, `data.negative_class` en el YAML.
- **6b. Cegamiento del desenlace.** [¿El evaluador del diagnóstico conocía el EEG? Normalmente sí es independiente; confirmar.]
- **7a. Predictores.** 114 valores de potencia absoluta por banda (delta, theta, alfa, beta, beta alta, gamma) × 19 canales y 1026 coherencias por par y banda, tal como vienen en la fuente. Derivados por este repositorio: potencia relativa, potencia regional, asimetría interhemisférica, coherencia regional (`features.*` del YAML; tabla `outputs/03_feature_meta.csv`).
- **7b. Cegamiento de predictores.** Características calculadas automáticamente; no hubo intervención manual.
- **8. Tamaño muestral.** 294 registros. Justificar como muestra de conveniencia; estimar eventos por parámetro candidato (95 controles como clase minoritaria frente a 242 características ⇒ alto riesgo de sobreajuste, mitigado con selección de características dentro de cada partición y modelos regularizados). Reportar la amplitud de los IC como evidencia de la imprecisión.
- **9. Datos faltantes.** Ninguna celda faltante en predictores (`outputs/01_inspection.json`). La columna vacía `Unnamed: 122` se eliminó. Sin variables demográficas disponibles.
- **10a. Manejo de predictores.** Transformaciones (relativización, log, asimetría) descritas en `src/eegdep/features.py` y parametrizadas en el YAML.
- **10b. Desarrollo del modelo.** Pipeline por partición: escalado → selección univariada de k características → clasificador; k y los hiperparámetros se eligen por CV interna agrupada. Calibración [sigmoide/isotónica] con CV interna. Todo ajustado solo con datos de entrenamiento de cada partición.
- **10c. Validación.** Holdout de ~20 % de sujetos (estratificado por clase) nunca usado durante el desarrollo. En el 80 % restante, CV externa `StratifiedGroupKFold` [n_splits × n_repeats] agrupada por sujeto. IC95 por bootstrap sobre sujetos. Prueba de permutación de etiquetas a nivel sujeto para la hipótesis nula de AUC = 0.5.
- **10d. Métricas.** AUC, sensibilidad, especificidad, F1, exactitud balanceada (umbral `validation.threshold`), Brier y ECE para calibración, curvas de calibración (`outputs/figures/calibration_*.png`).
- **10e. Comparación de modelos.** SVM lineal, SVM RBF, Random Forest, regresión logística L2 con el mismo esquema de validación; se declara el criterio de elección a priori (AUC en CV, no en holdout).
- **11. Interpretabilidad.** Efectos univariados topográficos calculados solo en desarrollo; importancia nativa y por permutación en holdout (`outputs/05_*`). Verificación de plausibilidad neurofisiológica frente a artefactos (p. ej. gamma frontal/temporal ⇒ sospecha de EMG; delta frontopolar ⇒ sospecha ocular).
- **12. Equidad.** No es posible evaluar rendimiento por edad, sexo u otros subgrupos porque el archivo no los contiene. Declarar como limitación mayor para una aplicación pública.
- **13. Participación de pacientes y público.** [Describir o declarar que no hubo.]

### Ética y datos
- **14. Aprobación ética y consentimiento.** [Comité de ética de la fuente, número, consentimiento para reuso secundario, licencia de la base, condiciones de redistribución.]
- **15. Disponibilidad de datos y código.** Código: este repositorio con `config/config.yaml` y semillas fijas. Datos: según licencia de la fuente.

## Resultados
- **16. Participantes.** Flujo: 294 filas de entrada → [n] tras QC (`outputs/02_qc_report.json`: duplicados, filas que fallan QC y política aplicada) → [n_dev] desarrollo / [n_test] holdout (`outputs/04_split.json`).
- **17. Desarrollo del modelo.** Hiperparámetros seleccionados por partición (`04_results.json: best_params_per_fold`), estabilidad de la selección de características.
- **18. Rendimiento.** Tabla por modelo: CV (media ± DE entre repeticiones; IC95 bootstrap) y holdout (puntual + IC95). Curvas de calibración. Resultado de la prueba de permutación. Advertir explícitamente si el holdout (≈59 sujetos) tiene IC más anchos que la CV.
- **19. Actualización del modelo.** No aplica.
- **20. Interpretación de características.** Topomapas por banda; discusión sobre coincidencia con literatura (asimetría alfa frontal, theta/alfa parietal, coherencia) y sobre señales sospechosas de artefacto.

## Discusión
- **21. Limitaciones.** Sin identificador de sujeto (asunción 1 fila = 1 sujeto); sin edad/sexo/medicación/comorbilidad; un solo sitio, un equipo, una población; características precalculadas con parámetros desconocidos (frecuencia de muestreo, referencia, condición ocular, duración); desbalance de clases; posible espectro de casos (depresión clínica vs. control sano sobreestima el rendimiento frente al escenario real de tamizaje con casos subclínicos y otros trastornos).
- **22. Interpretación.** Qué significa el AUC obtenido para tamizaje a la prevalencia real (VPP/VPN a prevalencias plausibles).
- **23. Implicaciones.** Pasos antes de cualquier uso público: validación externa multicéntrica, adquisición propia con protocolo documentado, estudio prospectivo, evaluación como software como dispositivo médico.

## Otra información
- **24. Registro.** [Protocolo registrado, si existe.]
- **25. Protocolo.** [Enlace.]
- **26. Financiación y conflictos.** [Declarar.]
- **27. Disponibilidad de modelo y supuestos de despliegue.** Cómo se obtiene la probabilidad, umbral y advertencia de que no es un diagnóstico.
