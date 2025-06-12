import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Order Date': pd.date_range(start='2021-01-01', periods=100),
    'Sales': np.random.randint(100, 500, size=100)
})
data = data.sort_values('Order Date')

def create_lagged_features(data, lag=1):
    lagged_data = data.copy()
    for i in range(1, lag + 1):
        lagged_data[f'lag_{i}'] = lagged_data['Sales'].shift(i)
    return lagged_data

lag = 5
sales_with_lags = create_lagged_features(data[['Order Date', 'Sales']], lag)


sales_with_lags = sales_with_lags.dropna()


X = sales_with_lags.drop(columns=['Order Date', 'Sales'])
y = sales_with_lags['Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model_xgb.fit(X_train, y_train)


predictions_xgb = model_xgb.predict(X_test)


rmse_xgb = np.sqrt(mean_squared_error(y_test, predictions_xgb))
print(f"RMSE: {rmse_xgb:.2f}")


plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Sales', color='red')
plt.plot(y_test.index, predictions_xgb, label='Predicted Sales', color='green')
plt.title('Sales Forecasting using XGBoost')
plt.xlabel('Index')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
