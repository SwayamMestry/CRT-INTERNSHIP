#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


np.random.seed(42)
n_samples = 100
prices = np.random.uniform(10, 100, n_samples)
sales = 1000 - 10 * prices + np.random.normal(0, 50, n_samples)


# In[3]:


data = pd.DataFrame({'Price': prices, 'Sales': sales})


# In[4]:


plt.figure(figsize=(10, 6))
plt.scatter(data['Price'], data['Sales'])
plt.title('Pricing and Sales Data')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.show()


# In[5]:


X = data[['Price']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[7]:


y_pred = model.predict(X_test)


# In[8]:


r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


# In[9]:


plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Linear Regression Model for Price Optimization')
plt.xlabel('Price')
plt.ylabel('Sales')
plt.legend()
plt.show()

