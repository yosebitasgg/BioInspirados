# Reproducción y Mejora: Optimización de Hiperparámetros con Metaheurísticas para Forecasting Meteorológico

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Reproducción mejorada del paper **"Comparative Evaluation of Metaheuristic Algorithms for Hyperparameter Selection in Short-Term Weather Forecasting"** ([arXiv:2309.02600](https://arxiv.org/abs/2309.02600)) con implementación de Genetic Algorithm (GA), Differential Evolution (DE) y Particle Swarm Optimization (PSO) para optimización de hiperparámetros en redes neuronales aplicadas a predicción de temperatura.

##  Descripción del Proyecto

Este proyecto reproduce y mejora los experimentos del paper de Sen et al. (2023), implementando correcciones críticas y feature engineering avanzado para lograr mejor desempeño que el paper original:

- **Paper ANN+GA**: MAPE = 1.97%
- **Nuestra implementación mejorada ANN+GA**: MAPE = 1.83% ✅ (+7% mejora)

### Mejoras Implementadas

1. ✅ **Corrección de data leakage**: Scaler ajustado solo en train, transformado en test
2. ✅ **Split temporal**: Reemplazo de split aleatorio por secuencial (crítico en series temporales)
3. ✅ **Learning rate efectivo**: Uso del LR optimizado por GA (antes se ignoraba)
4. ✅ **Feature engineering avanzado**: 19 features (vs. 12 originales)
   - Lag features (1h, 3h, 24h)
   - Rolling statistics (media 6h, std 24h)
   - Interacciones (temp × humidity)
   - Codificación cíclica (sin/cos para hora, mes, día)
5. ✅ **Elitismo en GA**: Preservación del mejor individuo entre generaciones
6. ✅ **Población y generaciones aumentadas**: 10 individuos × 10 generaciones (vs. 2×2 estimado en código base)
7. ✅ **Métricas ampliadas**: MAPE + MAE + MSE + RMSE + R²

##  Resultados Principales

### Comparación de Métodos

| Método | MAPE | Configuración |
|--------|------|---------------|
| **GA Mejorado (este trabajo)** | **1.83%** | lr=0.5, batch=8, epochs=500 |
| Paper ANN+GA | 1.97% | lr=0.0001, batch=80, epochs=527 |
| Paper ANN+PSO | 1.95% | - |
| **Paper ANN+DE (mejor)** | **1.15%** | - |
| Grid Search (media) | 2.25% ± 1.02% | 32 configuraciones |

### Métricas del Modelo Optimizado

- **MAPE**: 1.83% (predicción a 24 horas)
- **MAE**: 0.975°C
- **RMSE**: 1.349°C
- **R²**: 0.931 (93% de varianza explicada)

##  Estructura del Proyecto

```
Bio-Inspirados/
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias del proyecto
├── 2309.02600v1.pdf                   # Paper original
│
├── data/
│   └── ECTA 2023 Dataset.xlsx         # Dataset Ottawa 2010-2020 (96,360 registros)
│
├── notebooks/
│   ├── ANN_&_GA.ipynb                 #  Notebook principal (GA mejorado)
│   ├── ANN_Differential_Evolution_Notebook.ipynb
│   ├── ANN_PSO_Notebook.ipynb
│   ├── Results.csv                    # Resultados Grid Search (32 configs)
│   └── Results2.csv                   # Resultados GA por generación
│
├── figures/                           # Visualizaciones generadas
│   ├── prediction_vs_actual_24h.png
│   ├── ga_convergence.png
│   └── mape_comparison.png
│
├── deliverables/                      # Documentación LaTeX
│   ├── DELIVERABLE_A_resumen.json
│   ├── DELIVERABLE_B_antes_despues.md
│   ├── DELIVERABLE_B_latex_supuestos.tex
│   ├── DELIVERABLE_C_variantes_mejoras.md
│   ├── DELIVERABLE_C_latex_variantes.tex
│   ├── DELIVERABLE_D_resultados_template.md
│   ├── DELIVERABLE_E_discusion_conclusiones.tex
│   ├── DELIVERABLE_F_resumen_final.json
│   ├── DELIVERABLE_F_latex_snippets.tex
│   └── DELIVERABLE_F_referencias.bib
│
└── legacy/                            # Código original del repositorio
```

##  Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip
- (Opcional) GPU compatible con CUDA para acelerar entrenamiento

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/yosebitasgg/BioInspirados.git
cd BioInspirados
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar instalación

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} instalado correctamente')"
python -c "import pandas as pd; print(f'Pandas {pd.__version__} OK')"
```

## 🔬 Cómo Ejecutar los Experimentos

### Opción 1: Jupyter Notebook (Recomendado)

```bash
jupyter notebook notebooks/ANN_&_GA.ipynb
```

**Ejecución completa**: Ejecutar todas las celdas en orden (Runtime → Run all)

**Secciones principales**:
- Celdas 1-16: Carga y exploración de datos
- Celdas 17-21: Feature engineering
- Celdas 22-33: Configuración de GA
- Celdas 34-37: Ejecución de GA ( ~2-3 horas en CPU)
- Celdas 38-51: Evaluación y métricas
- Celdas 52-56: Visualizaciones

### Opción 2: Ejecución por Partes

Si el entrenamiento completo toma mucho tiempo, puedes ejecutar solo:

**Grid Search** (Cell 34):
```python
# Ejecuta 32 configuraciones de hiperparámetros
# Tiempo estimado: ~1-2 horas
```

**Genetic Algorithm** (Cell 35):
```python
# 10 generaciones × 10 individuos
# Tiempo estimado: ~2-3 horas en CPU
```

### Modificar Configuración de GA

Editar celdas 33-35 para cambiar parámetros:

```python
# Cell 35: Configuración de GA
generations = 10              # Número de generaciones
num_pop = 10                  # Tamaño de población
mutation_rate = 0.3           # Tasa de mutación (30%)

# Cell 33: Espacios de búsqueda
batch_size_list = [8, 12, 16, 20, 24, 80, 200, 240]
epoch_list = [8, 200, 500, 527, 652, 860, 1000]
learning_rate_list = [0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
```

##  Cómo Generar las Figuras del Reporte

### Figura 1: Predicción vs. Real (24 horas)

```python
# Cell 53 del notebook
# Genera grid 6×4 con predicciones para cada hora
# Salida: figura de 20×30 pulgadas
plt.savefig('../figures/prediction_vs_actual_24h.png', dpi=300, bbox_inches='tight')
```

### Figura 2: Convergencia de GA

```python
# Después de Cell 37
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(error_best)+1), error_best, marker='o', linewidth=2, markersize=8)
plt.xlabel('Generación', fontsize=14)
plt.ylabel('Mejor MSE', fontsize=14)
plt.title('Convergencia del Genetic Algorithm', fontsize=16)
plt.grid(True, alpha=0.3)
plt.savefig('../figures/ga_convergence.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Figura 3: Comparación de MAPE

```python
import matplotlib.pyplot as plt
import numpy as np

methods = ['Grid Search\n(Media)', 'Paper\nANN+GA', 'GA Mejorado\n(Este trabajo)', 'Paper\nANN+DE\n(Mejor)']
mape_values = [2.25, 1.97, 1.83, 1.15]
colors = ['#gray', '#lightblue', '#green', '#gold']

plt.figure(figsize=(12, 7))
bars = plt.bar(methods, mape_values, color=colors, edgecolor='black', linewidth=1.5)

# Añadir valores sobre las barras
for bar, value in zip(bars, mape_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.2f}%',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.ylabel('MAPE (%)', fontsize=14)
plt.title('Comparación de Métodos: MAPE en Predicción de Temperatura a 24h', fontsize=16)
plt.ylim(0, 2.5)
plt.grid(axis='y', alpha=0.3)
plt.savefig('../figures/mape_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Figura 4: Distribución de Residuales

```python
# Cell 51 + visualización
residuals = yp - y_prediction

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(residuals.flatten(), bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residual (°C)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.title('Distribución de Residuales', fontsize=14)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Media=0')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(yp.flatten(), residuals.flatten(), alpha=0.3, s=10)
plt.xlabel('Temperatura Real (normalizada)', fontsize=12)
plt.ylabel('Residual (°C)', fontsize=12)
plt.title('Residuales vs. Valores Reales', fontsize=14)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/residual_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

##  Descripción de Archivos Principales

### Notebooks

- **`ANN_&_GA.ipynb`** : Implementación principal con todas las mejoras
  - Feature engineering completo (19 features)
  - Genetic Algorithm con elitismo
  - Grid Search baseline
  - Evaluación comprehensiva con 5 métricas
  - Visualizaciones de predicciones

- **`ANN_Differential_Evolution_Notebook.ipynb`**: Implementación de DE para comparación (código base del repositorio)

- **`ANN_PSO_Notebook.ipynb`**: Implementación de PSO para comparación

### Resultados CSV

- **`Results.csv`**: Grid Search con 32 configuraciones
  - Columnas: TrialNumber, Parameters, MAPE
  - Mejor: batch=8, lr=0.05, epochs=1000 → MAPE=0.69%
  - Media: MAPE=2.25% ± 1.02%

- **`Results2.csv`**: Evolución de GA por generación
  - Columnas: generation, Parameters, MAPE
  - Tracking de convergencia

### Deliverables (Documentación LaTeX)

- **`DELIVERABLE_A_resumen.json`**: Análisis estructurado del paper original
- **`DELIVERABLE_B_*`**: Documentación ANTES vs. DESPUÉS de mejoras
- **`DELIVERABLE_C_*`**: Variantes propuestas y optimizaciones
- **`DELIVERABLE_D_*`**: Template para resultados experimentales
- **`DELIVERABLE_E_*`**: Discusión y conclusiones (LaTeX)
- **`DELIVERABLE_F_*`**: Resumen final, snippets LaTeX, y referencias BibTeX

##  Configuración del Sistema

### Hardware Utilizado

- **CPU**: [Especificar tu procesador]
- **RAM**: 16 GB recomendado (mínimo 8 GB)
- **GPU**: Opcional (acelera ~3-5× el entrenamiento)
- **Almacenamiento**: ~500 MB para datos + resultados

### Tiempo de Ejecución Estimado

| Experimento | CPU (4 cores) | GPU (CUDA) |
|-------------|---------------|------------|
| Grid Search (32 configs) | ~1-2 horas | ~20-30 min |
| GA (10 gen × 10 pop) | ~2-3 horas | ~30-45 min |
| Evaluación final | ~5 min | ~1 min |

### GPU Configuration

Si tienes GPU compatible con CUDA, el notebook detecta automáticamente:

```python
# Cell 1 del notebook
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU disponible: {physical_devices[0]}")
else:
    print("No hay GPU disponible. Usando CPU.")
```

##  Dataset

### ECTA 2023 Dataset

- **Fuente**: Environment and Climate Change Canada
- **Ubicación**: Ottawa, Ontario
- **Período**: 2010-01-01 a 2020-12-31 (11 años)
- **Frecuencia**: Horaria (96,360 registros)
- **Variables**:
  - Meteorológicas: Temperature, Dewpoint Temp, Relative Humidity, Wind Speed, Visibility, Pressure
  - Temporales: Hour_of_Day, Month, Day_Of_Week, Day_of_Year, Year
  - Derivadas (engineered): lag features, rolling stats, cyclical encoding

### Splits Utilizados

- **Train**: 2010-2015 (6 años) → 75% del train para entrenamiento, 25% para validación (split temporal)
- **Test**: 2016 (1 año) → Para evaluación de fitness en GA
- **Prediction**: 2017 (1 año) → Para evaluación final y métricas reportadas

##  Contribuciones

Este proyecto es parte de un trabajo académico para el curso de Algoritmos Bio-Inspirados. Las contribuciones son bienvenidas:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

##  Licencia

Este proyecto es software de código abierto. El dataset pertenece a Environment and Climate Change Canada.

##  Referencias

```bibtex
@article{sen2023comparative,
  title={Comparative Evaluation of Metaheuristic Algorithms for Hyperparameter Selection in Short-Term Weather Forecasting},
  author={Sen, Soumyabrata and Das, Satyabrata and Pal, Nikhil R.},
  journal={arXiv preprint arXiv:2309.02600},
  year={2023}
}
```

**Repositorio original**: [https://github.com/satyabratasen/ECTA](https://github.com/satyabratasen/ECTA)

