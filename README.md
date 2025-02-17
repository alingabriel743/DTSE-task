# DTSE-task
This project aims to forecast the quarterly unemployment rate using macroeconomic indicators such as real GDP, population, and inflation, covering the period from 1959Q1 to 2013Q4. The focus was on building multiple models, including SARIMAX, VECM, Random Forest, and LGBM Regressor, to capture both short-term and long-term patterns. Additionally, an ensemble model combining predictions from the statistical models was developed.

The data processing represents the first step. Missing values were identified and handled appropriately, and all time series were tested for stationarity using the ADF (Augmented Dickey-Fuller) test. Non-stationary variables such as realgdp, pop, and unemp_wins were differenced to ensure stationarity. Additionally, new features were engineered, including infl_rolling_mean (rolling average of inflation) and pop_diff_lag_1 (one-period lag of population difference) to capture short-term patterns and economic dependencies (also being tested for stationarity).
Four models were then built and evaluated:

- SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous Regressors):
The SARIMAX model was chosen to capture short-term dependencies and account for external influences (exogenous variables). After analyzing ACF and PACF plots, an order of (1,0,1) was selected.

- VECM (Vector Error Correction Model):
The VECM model was chosen to capture long-term relationships between macroeconomic indicators. Using the Johansen cointegration test, a cointegration rank of 4 was selected, indicating strong long-term dependencies.

- Random Forest Regressor:
A machine learning model was included to capture non-linear relationships. TimeSeriesSplit cross-validation was performed with 5 folds.

- LGBM Regressor (Light Gradient Boosting Machine):
LGBM, a gradient-boosting algorithm optimized for speed and efficiency, was trained with the same cross-validation approach as Random Forest.

The notebooks inside this project are:
1. N1_STAT_DTSE_AIS_unemployment_task - includes the EDA and the statistical modeling approaches
2. N2_ML_DTSE_AIS_unemployment_task - containing the ML models developed for experimenting on the time series present in the dataset
3. utils.py - some miscellaneous functions I did not wanted to include in the notebooks in order to keep them clean
