# AI-Powered Sales Forecasting System

A Streamlit-based machine learning application that analyzes historical sales data, forecasts future sales, simulates business scenarios, and provides data-backed chatbot insights.

## Overview

This project helps convert raw sales datasets into useful business insights. It includes data cleaning, exploratory analysis, forecasting model comparison, scenario simulation, and an AI-style chatbot that answers questions based on the uploaded dataset.

## Key Features

| Module | Description |
| --- | --- |
| Data Upload | Upload CSV or Excel files, validate required columns, and clean data before analysis |
| Sales Analysis | Explore sales trends, category performance, regional performance, anomalies, and visual insights |
| Forecasting | Compare forecasting models and generate future sales predictions |
| Scenario Simulation | Test assumptions such as growth rate, discounts, marketing impact, churn, and new sales channels |
| Chatbot Insights | Ask dataset-based questions about trends, categories, regions, anomalies, and forecasts |

## Machine Learning and Forecasting

The forecasting module compares multiple approaches and selects the best model using validation metrics.

Models used:

- Moving Average
- Linear Trend
- Seasonal Trend
- Seasonal Naive baseline with growth adjustment
- Exponential Smoothing
- Holt's Double Smoothing
- Random Forest
- Smart Ensemble

Model performance is evaluated using:

- MAE
- RMSE
- MAPE (%)
- R-squared

On the built-in sample dataset, the current best result is:

- Best model: Exponential Smoothing
- MAPE: 7.29%
- Approximate accuracy: 92.71%

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- Requests
- OpenPyXL
- Pytest

## Project Structure

```text
AI-3/
|-- app/
|   |-- app.py
|   |-- pages/
|   |   |-- 1_Data_Upload.py
|   |   |-- 2_Analysis.py
|   |   |-- 3_Forecasting.py
|   |   |-- 4_Scenario_Simulation.py
|   |   `-- 5_Chatbot.py
|-- services/
|   |-- chatbot_service.py
|   |-- evaluation_service.py
|   |-- forecasting_service.py
|   |-- insight_service.py
|   |-- model_service.py
|   |-- preprocessing_service.py
|   `-- simulation_service.py
|-- data/
|   `-- raw/
|       `-- sample_sales.csv
|-- notebooks/
|   `-- model_experiments.ipynb
|-- tests/
|-- utils/
|-- requirements.txt
|-- Dockerfile
|-- .gitignore
`-- README.md
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-sales-forecasting.git
cd ai-sales-forecasting
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app/app.py
```

Then open:

```text
http://localhost:8501
```

## Testing

Run the test suite with:

```bash
pytest
```

## Chatbot Notes

The chatbot supports direct, data-backed answers from the uploaded dataset. It can answer questions such as:

- What is the overall sales summary?
- Which category performed best?
- Which region has the highest sales?
- Are there any sales anomalies?
- What is the forecast for the next 6 months?
- What is the category-wise forecast?

Optional Gemini-based explanations can be enabled by adding your API key in Streamlit secrets. Do not commit secrets to GitHub.

## Deployment

This project can be deployed on Streamlit Community Cloud:

1. Push the project to GitHub.
2. Create a new Streamlit Community Cloud app.
3. Select the GitHub repository.
4. Set the main file path to:

```text
app/app.py
```

5. Add secrets only if chatbot explanation features require an API key.

## Resume Description

Developed an AI-powered sales forecasting system using Python, Streamlit, and machine learning to predict future sales from historical data. Performed data preprocessing, exploratory data analysis, feature engineering, model comparison, and forecast evaluation using metrics such as MAE, RMSE, MAPE, and R-squared. Built interactive dashboards and chatbot-based insights to support sales planning, inventory decisions, and business recommendations.

## License

This project is licensed under the MIT License.
