from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
import numpy as np

def adf_test(series, column_name):
    """
    Perform Augmented Dickey-Fuller test on a time series
    Args:
        series: time series to test
        column_name: name of the column
    Returns:
        None
    """
    result = adfuller(series, autolag='AIC')
    print(f'ADF test for {column_name}:')
    print(f'ADF statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical values:')
    for key, value in result[4].items():
        print(f'{key}: {value}')
    print(f'Stationary: {result[1] < 0.05}\n')

def make_stationary(series, max_diff=5):
    """
    Make a time series stationary by differencing
    Args:
        series: time series to difference
        max_diff: maximum number of differencing allowed
    returns:
        series: differenced series
    """
    for i in range(max_diff):
        result = adfuller(series, autolag='AIC')
        print(f'Order {i}: ADF statistic={result[0]}, p-value={result[1]}')
        if result[1] <= 0.05:
            print(f'Series is stationary after {i} diffs')
            return series
        series = series.diff().dropna()
    print('Series did not become stationary within the specified differencing limit')
    return series

def granger_causality_matrix(data, variables, max_lag=5):
    """
    
    """
    matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for col in matrix.columns:
        for row in matrix.index:
            if row != col:
                test_result = grangercausalitytests(data[[row, col]], max_lag, verbose=False)
                p_values = [round(test_result[i+1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]
                matrix.loc[row, col] = min(p_values)  
            else:
                matrix.loc[row, col] = 1 
    return matrix

def calculate_dof(data, max_lag):
    n_obs = len(data)
    n_vars = len(data.columns)
    dof = n_obs - (n_vars * max_lag + 1)
    return dof