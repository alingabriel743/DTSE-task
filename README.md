# DTSE-task
This project aims to forecast the quarterly unemployment rate using macroeconomic indicators such as real GDP, population, and inflation, covering the period from 1959Q1 to 2013Q4. The focus was on building multiple models, including SARIMAX, VECM, Random Forest, and LGBM Regressor, to capture both short-term and long-term patterns. Additionally, an ensemble model combining predictions from the statistical models was developed.

The first step is represented by the data processing. Missing values were identified and handled appropriately, and all time series were tested for stationarity using the ADF (Augmented Dickey-Fuller) test. Non-stationary variables such as realgdp, pop, and unemp_wins were differenced to ensure stationarity. Additionally, new features were engineered, including infl_rolling_mean (rolling average of inflation) and pop_diff_lag_1 (one-period lag of population difference) to capture short-term patterns and economic dependencies (also being tested for stationarity).
Four models were then built and evaluated:

- SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous Regressors):
The SARIMAX model was chosen to capture short-term dependencies and account for external influences (exogenous variables). After analyzing ACF and PACF plots, an order of (1,0,1) was selected.

- VECM (Vector Error Correction Model):
The VECM model was chosen to capture long-term relationships between macroeconomic indicators. Using the Johansen cointegration test, a cointegration rank of 4 was selected, indicating strong long-term dependencies.

- Random Forest Regressor:
A machine learning model was included to capture non-linear relationships. TimeSeriesSplit cross-validation was performed with 5 folds.

- LGBM Regressor (Light Gradient Boosting Machine):
LGBM, a gradient-boosting algorithm optimized for speed and efficiency, was trained with the same cross-validation approach as Random Forest.
