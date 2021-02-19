#!/usr/bin/env python
# coding: utf-8

# # Import libraries

import pandas as pd
import datetime as dt

data = pd.read_csv(r'C:/Users/bicers/gccr003_data.csv',sep=';',encoding='unicode_escape')
data = data.dropna()
data["Date_of_onset"] = pd.to_datetime(data["Date_of_onset"]) 
data["Date_of_onset_Time_Stamp"] = pd.to_datetime(data["Date_of_onset_Time_Stamp"])
data["Smell_before_illness_Time_Stamp"] = pd.to_datetime(data["Smell_before_illness_Time_Stamp"])
data["Smell_during_illness_Time_Stamp"] = pd.to_datetime(data["Smell_during_illness_Time_Stamp"])
data["Taste_before_illness_Time_Stamp"] = pd.to_datetime(data["Taste_before_illness_Time_Stamp"])
data["Taste_during_illness_Time_Stamp"] = pd.to_datetime(data["Taste_during_illness_Time_Stamp"])
data["Smell_after_illness_Time_Stamp"] = pd.to_datetime(data["Smell_after_illness_Time_Stamp"])
data["Taste_after_illness_Time_Stamp"] = pd.to_datetime(data["Taste_after_illness_Time_Stamp"])
data["Smell_current_Time_Stamp"] = pd.to_datetime(data["Smell_current_Time_Stamp"])
data["Smell_most_impaired_Time_Stamp"] = pd.to_datetime(data["Smell_most_impaired_Time_Stamp"])
data["Taste_current_Time_Stamp"] = pd.to_datetime(data["Taste_current_Time_Stamp"])
data["Taste_most_impaired_Time_Stamp"] = pd.to_datetime(data["Taste_most_impaired_Time_Stamp"])
data.info()
data.describe()
data.head()
data['percentage_recovery_smell'] = data['Smell_current'] / data['Smell_before_illness'] * 100
data['percentage_recovery_taste'] = data['Taste_current'] / data['Taste_before_illness'] * 100
data['percentage_recovery_smell'].head()
len(data.query('percentage_recovery_smell < 80'))
len(data.query('percentage_recovery_smell > 100'))
len(data.query('percentage_recovery_taste < 80'))
data['Date_of_onset'] = pd.to_datetime(data['Date_of_onset'])
data['days_from_onset'] = ((data['Smell_after_illness_Time_Stamp'] - data['Date_of_onset']).dt.total_seconds()/(60*60*24))
data['days_from_onset'].head()
X = data[['days_from_onset']]
y = data['Smell_before_illness'] - data['Smell_current']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
import matplotlib.pyplot as plt
print('Coefficients: \n', lm.coef_)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
