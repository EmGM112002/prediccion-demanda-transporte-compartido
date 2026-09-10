# Predicción de Demanda de Transporte Compartido mediante Random Forest

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Área](https://img.shields.io/badge/Área-Analítica%20Predictiva-1B3A6B?style=flat)
![Modelo](https://img.shields.io/badge/Modelo-Random%20Forest-2E6DB4?style=flat)

Proyecto de Machine Learning desarrollado en **Python** para predecir la demanda horaria de viajes de transporte bajo demanda en Nueva York.

El proyecto transforma registros individuales de viajes en una serie de demanda agregada por hora y utiliza variables temporales para entrenar un modelo **Random Forest Regressor**.

Aunque no corresponde directamente a un problema actuarial, el proyecto demuestra competencias de modelación predictiva, preparación de datos, validación y evaluación fuera de muestra aplicables a problemas cuantitativos.

El desarrollo conceptual del algoritmo se encuentra en el documento técnico incluido en el repositorio.

---

## 1. Problema analítico

La demanda de servicios de movilidad presenta variaciones importantes de acuerdo con el momento del día y el calendario.

Estas variaciones pueden afectar:

- disponibilidad de vehículos;
- tiempos de espera;
- asignación de recursos;
- capacidad operativa;
- y planificación de flota.

El problema consiste en predecir el **número de viajes que ocurrirán durante una hora** utilizando características temporales derivadas de los registros de viajes.

El objetivo es construir un flujo completo que transforme observaciones transaccionales en una variable agregada de demanda y posteriormente evalúe la capacidad de un algoritmo no lineal para predecirla.

---

## 2. Metodología

El proyecto utiliza un **Random Forest Regressor**.

El flujo implementado es:

1. Cargar los archivos de entrenamiento y prueba.
2. Convertir `pickup_datetime` a formato de fecha y hora.
3. Redondear cada registro al inicio de su hora correspondiente.
4. Agrupar los viajes por hora.
5. Contar el número de viajes dentro de cada intervalo.
6. Generar variables temporales.
7. Separar predictores y variable objetivo.
8. Construir un Random Forest base.
9. Evaluar diferentes combinaciones de hiperparámetros mediante `GridSearchCV`.
10. Seleccionar el mejor estimador.
11. Generar predicciones sobre el conjunto de prueba.
12. Evaluar mediante MAE, RMSE y \(R^2\).
13. Comparar visualmente valores observados y predichos.

### Variables predictoras

El modelo utiliza principalmente:

- hora del día;
- día de la semana;
- mes.

### Optimización de hiperparámetros

Se utiliza `GridSearchCV` con validación cruzada de **3 folds**.

La combinación seleccionada por la búsqueda fue:

| Hiperparámetro | Valor |
|---|---:|
| `n_estimators` | 200 |
| `max_depth` | 10 |
| `min_samples_split` | 10 |

El modelo utiliza `random_state = 42` para hacer reproducible el componente aleatorio del algoritmo.

---

## 3. Datos utilizados

El notebook trabaja con dos archivos:

```text
data/train.csv
data/test.csv
```

Los registros contienen una variable temporal denominada:

```text
pickup_datetime
```

que se utiliza para transformar la información transaccional en conteos horarios de viajes.

La lógica de preparación convierte cada observación individual en una marca temporal redondeada a la hora y posteriormente agrega el número de registros dentro de cada intervalo.

Los archivos de datos **no se encuentran actualmente incluidos en el repositorio**, por lo que deben añadirse localmente dentro de la carpeta `data/` antes de ejecutar el notebook.

---

## 4. Herramientas

| Herramienta | Aplicación |
|---|---|
| **Python** | Desarrollo completo del flujo de Machine Learning |
| **pandas** | Transformación de registros y agregación temporal |
| **NumPy** | Operaciones numéricas |
| **scikit-learn** | Entrenamiento, optimización y evaluación del modelo |
| **RandomForestRegressor** | Predicción de demanda |
| **GridSearchCV** | Selección de hiperparámetros |
| **Matplotlib** | Visualización de resultados |
| **Seaborn** | Visualización exploratoria |
| **Jupyter Notebook** | Desarrollo reproducible |

---

## 5. Resultados

El mejor modelo seleccionado utiliza:

```text
n_estimators = 200
max_depth = 10
min_samples_split = 10
```

La evaluación sobre el conjunto de prueba produce:

| Métrica | Resultado |
|---|---:|
| MAE | 14.79 viajes |
| RMSE | 21.32 viajes |
| R² | 0.8953 |

### Interpretación

El **MAE de 14.79** indica que el error absoluto medio de las predicciones es de aproximadamente 15 viajes por intervalo horario.

El **RMSE de 21.32** penaliza con mayor intensidad los errores grandes y permite complementar la interpretación del MAE.

El valor:

**R² = 0.8953**

indica que, sobre el conjunto de prueba utilizado en este proyecto, el modelo explica aproximadamente **89.5% de la variabilidad observada en la demanda horaria**.

Este resultado describe únicamente el desempeño obtenido sobre los datos y la partición utilizada en el proyecto. No garantiza que el mismo desempeño se mantenga ante periodos, ciudades o patrones de movilidad diferentes.

---

## 6. Aprendizajes y limitaciones

### Aprendizajes

El proyecto permite demostrar:

- transformación de registros transaccionales en información temporal;
- agregación de datos por hora;
- ingeniería de variables de calendario;
- entrenamiento de modelos de Machine Learning;
- utilización de Random Forest para regresión;
- búsqueda de hiperparámetros;
- validación cruzada;
- evaluación fuera de muestra;
- interpretación de MAE, RMSE y R²;
- y visualización de valores observados frente a predicciones.

Además, demuestra cómo un problema operativo puede reformularse como un problema de regresión supervisada.

### Limitaciones

Entre las principales limitaciones del proyecto se encuentran:

- utiliza únicamente características temporales relativamente simples;
- no incorpora variables meteorológicas;
- no incorpora eventos especiales;
- no incorpora información geográfica;
- no utiliza rezagos explícitos de demanda;
- no modela tendencias mediante un modelo específico de series temporales;
- `GridSearchCV` utiliza validación cruzada estándar y no un esquema diseñado específicamente para preservar el orden temporal;
- en un problema de pronóstico estrictamente temporal sería recomendable comparar con `TimeSeriesSplit` o validación walk-forward;
- el algoritmo no permite extrapolar fácilmente patrones que no hayan aparecido en los datos de entrenamiento;
- las métricas obtenidas dependen del conjunto de prueba utilizado;
- y los archivos `train.csv` y `test.csv` requeridos por el notebook no se encuentran actualmente versionados en el repositorio.

Como extensión, el proyecto podría compararse con modelos de series temporales, Gradient Boosting o modelos que incorporen variables meteorológicas, geográficas y rezagos de demanda.

---

## 7. Contenido del repositorio

```text
prediccion-demanda-transporte-compartido/
├── Codigo.ipynb
├── Proyecto_6___Predicción_de_Demanda___Modelo_random_forest.pdf
└── README.md
```

### Archivos requeridos localmente

Para ejecutar el notebook también se requiere crear:

```text
data/
├── train.csv
└── test.csv
```

Estos archivos son utilizados por el código, pero actualmente no forman parte del repositorio.

---

## 8. Documentación técnica

El desarrollo conceptual y la explicación matemática del algoritmo se encuentran en:

**[`Proyecto_6___Predicción_de_Demanda___Modelo_random_forest.pdf`](./Proyecto_6___Predicción_de_Demanda___Modelo_random_forest.pdf)**

La implementación completa se encuentra en:

**[`Codigo.ipynb`](./Codigo.ipynb)**

---

## 9. Cómo ejecutar el proyecto

### Requisitos

- Python 3;
- Jupyter Notebook, JupyterLab, Positron, VS Code o un entorno compatible con `.ipynb`.

### Dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Preparación de datos

Crear la siguiente estructura dentro del repositorio:

```text
data/
├── train.csv
└── test.csv
```

Los dos archivos deben contener la variable:

```text
pickup_datetime
```

esperada por el notebook.

### Ejecución

1. Clonar o descargar el repositorio.
2. Crear la carpeta `data`.
3. Colocar `train.csv` y `test.csv` en dicha carpeta.
4. Abrir `Codigo.ipynb`.
5. Ejecutar las celdas secuencialmente.
6. Revisar la transformación de los registros a demanda horaria.
7. Revisar la ingeniería de variables.
8. Ejecutar la búsqueda de hiperparámetros.
9. Analizar el mejor Random Forest obtenido.
10. Revisar MAE, RMSE y R².
11. Analizar las visualizaciones de valores reales y predichos.

---

## Autor

**Emiliano Guillén Medina**  
Licenciatura en Actuaría  
[GitHub](https://github.com/EmGM112002) · [LinkedIn](https://www.linkedin.com/in/emgm11)
