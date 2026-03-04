# Walmart Sales Forecasting (MLOps Lab)

У цьому проєкті виконано аналіз даних продажів Walmart та побудовано модель машинного навчання для прогнозування Weekly_Sales.

## Структура проєкту
mlops_lab_1/
├── notebooks/
│ └── 01_eda.ipynb
├── src/
│ └── train.py
├── requirements.txt
└── README.md

## Етапи роботи

1. Проведено EDA (аналіз даних) у `notebooks/01_eda.ipynb`
2. Реалізовано скрипт тренування моделі `src/train.py`
3. Проведено експерименти та логування результатів у MLflow

## Запуск навчання

```bash
python src/train.py

## Перегляд результатів
```bash
mlflow ui
За адресою: http://127.0.0.1:5000
