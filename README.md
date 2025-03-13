*ADSP 31006: Time Series Analysis and Forecasting – Final Project*

# Forecasting Air Quality Index in London using multiple advanced time series methods.

## Project Overview

Air pollution is a major concern in urban areas, particularly in London, where poor air quality contributes to health risks such as respiratory diseases and cardiovascular issues. The goal of this project is to develop an accurate time series forecasting model for predicting the Air Quality Index (AQI) in London, allowing stakeholders to take timely intervention measures.

Our approach involves evaluating multiple forecasting models, including traditional statistical models and deep learning techniques, to determine the most effective method for predicting AQI.

## Data

We used historical AQI data from **Open-Meteo**, spanning from **January 2, 2013, to February 28, 2025**. The dataset consists of hourly AQI readings for London, which were preprocessed by interpolating missing values and aggregating them into daily readings. The dataset was divided into:
- **Training Set:** January 2, 2013 – December 31, 2024
- **Test Set:** January 1, 2025 – February 28, 2025

## Models

We implemented and evaluated the following models:

1. **ARIMA, SARIMA, ARFIMA:**  
   - Traditional time series models capturing trends and seasonality.
   - Best performing: **First-order differenced ARFIMA** (RMSE = 18.68).

2. **Exponential Smoothing (ETS) & Holt-Winters:**  
   - Smoothing models used to forecast AQI trends.
   - Best performing: **First-order differenced ETS** (RMSE = 17.88).

3. **Bayesian Structural Time Series (BSTS):**  
   - A state-space model incorporating long-term trends and seasonality.
   - Best performing: **Seasonally differenced BSTS with AR(5)** (RMSE = 22.64).

4. **Long Short-Term Memory (LSTM):**  
   - A deep learning model that captures long-term dependencies in the AQI data.
   - **Best overall model:** LSTM with an RMSE of **10.2**, outperforming all statistical models.

## Key Findings

- Traditional time series models like ARFIMA and ETS provided reasonable forecasts but exhibited assumption violations such as autocorrelated residuals.
- BSTS improved upon these models by incorporating a Bayesian framework, but performance was still suboptimal.
- LSTM significantly outperformed all other models, demonstrating the effectiveness of deep learning for AQI forecasting.

## Future Improvements

- Fine-tuning LSTM hyperparameters (e.g., learning rate, number of layers).
- Incorporating external factors such as weather, traffic data, and emissions data.
- Exploring real-time forecasting capabilities for policy and intervention strategies.


## Authors

- **Anusha Bhat** - EDA, exponential smoothing models, report formatting  
- **Sara Chaker** - EDA, BSTS models, data transformations  
- **Hritik Jhaveri** - EDA, ARIMA/SARIMA/ARFIMA models, introduction  
- **John Melel** - EDA, LSTM models, GitHub repository, report formatting  

## Data Source

- [Open-Meteo API](https://open-meteo.com/en/docs/air-quality-api)  

## Citation

If you use this work, please cite:  
> Bhat, Melel, Chaker, Jhaveri (2025). *Forecasting Air Quality Index in London*.

