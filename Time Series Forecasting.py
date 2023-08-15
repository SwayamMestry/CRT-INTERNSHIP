#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error


# In[2]:


np.random.seed(42)
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.rand(len(date_rng)) * 100
data = pd.DataFrame({'Date': date_rng, 'Value': values})
data.set_index('Date', inplace=True)


# In[3]:


plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Original Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()


# In[4]:


train_size = int(0.8 * len(data))
train_data, test_data = data[:train_size], data[train_size:]


# In[5]:


model = auto_arima(train_data['Value'], seasonal=False, trace=True)


# In[6]:


model_fit = model.fit(train_data['Value'])


# In[7]:


forecast_steps = len(test_data)
forecast = model_fit.predict(n_periods=forecast_steps)


# In[8]:


forecast_dates = pd.date_range(start=test_data.index[0], periods=forecast_steps, freq='D')
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
forecast_df.set_index('Date', inplace=True)


# In[9]:


plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Train Data')
plt.plot(test_data, label='Test Data')
plt.plot(forecast_df, label='Forecast', linestyle='dashed')
plt.title('Time Series Forecasting with ARIMA')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()


# In[10]:


rmse = np.sqrt(mean_squared_error(test_data['Value'], forecast_df['Forecast']))
print(f"Root Mean Squared Error (RMSE): {rmse}")

