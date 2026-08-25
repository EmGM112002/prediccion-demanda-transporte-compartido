# Portafolio Analítico — Predicción de Demanda
## Modelo Random Forest para Volumen de Viajes en NYC

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=flat&logo=latex&logoColor=white)
![Área](https://img.shields.io/badge/Área-Movilidad%20Urbana-1B3A6B?style=flat)
![Modelo](https://img.shields.io/badge/Modelo-Random%20Forest-2E6DB4?style=flat)

---

## ¿De qué trata este proyecto?

Canalización (pipeline) de Machine Learning enfocada en pronosticar la **demanda horaria de viajes** de transporte bajo demanda en la ciudad de Nueva York. El objetivo es estimar el volumen de transacciones utilizando variables temporales para optimizar el enrutamiento predictivo (predictive dispatch) de flotas.

El modelo emplea un ensamblaje de **Random Forest Regressor**, el cual captura de forma no paramétrica las relaciones altamente no lineales y cíclicas de la estacionalidad horaria, agregando múltiples árboles mediante *Bootstrap Aggregating* para mitigar la varianza:

$$\hat{f}_{rf}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(x)$$

| Característica | Detalle |
|---|---|
| Algoritmo | Random Forest Regressor |
| Predictores | Hora del día, día de la semana, mes |
| Criterio de División | Minimización del Error Cuadrático Medio (MSE) |
| Validación | GridSearchCV con validación cruzada (3-fold) |
| Dataset | Registros agregados de la Taxi and Limousine Commission (TLC) |

---

## Estructura del repositorio

```text
Ride-Sharing-Demand-Prediction/
│
├── data/
│   └── train.csv                           # Dataset transaccional (Kaggle/TLC)
├── docs/
│   └── marco_teorico.md                    # Documentación matemática formal
├── notebooks/
│   └── demand_prediction.ipynb             # Notebook secuencial ejecutable
└── README.md
