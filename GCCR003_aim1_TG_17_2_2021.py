#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import pandas as pd
import datetime as dt


# # Import data

# In[2]:


data = pd.read_csv(r'C:/Users/bicers/gccr003_data.csv',sep=';',encoding='unicode_escape')

data = data.dropna()


# # Convert DateTime data types

# In[3]:


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


# # Pre-Process

# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.head()


# # Relevant column names:

# 
# 
# Date_of_onset ,
# 
# Smell_before_illness,
# 
# Blocked_nose_before_illness,
# 
# Taste_before_illness,
# 
# Chemethesis_before_illness,
# 
# FOLLOWUP
# 
# Smell_current
# 
# Taste_current
# 
# Email_Time_Stamp_y
# 

# # Percentages of recovery

# In[7]:


data['percentage_recovery_smell'] = data['Smell_current'] / data['Smell_before_illness'] * 100
data['percentage_recovery_taste'] = data['Taste_current'] / data['Taste_before_illness'] * 100


# In[8]:


data['percentage_recovery_smell'].head()


# # Long haulers count
# Number of individuals with changes (under 80%)
# 
# <u> smell </u>

# In[9]:


len(data.query('percentage_recovery_smell < 80'))


# In[10]:


# len(data.query('percentage_recovery_smell < 70'))


# In[11]:


len(data.query('percentage_recovery_smell > 100'))


# interestingly there are 486 cases with higher percentage than the beggining

# Number of individuals with changes (under 80%)
# 
# <u> taste </u>

# In[12]:


len(data.query('percentage_recovery_taste < 80'))


# In[13]:


# len(data.query('percentage_recovery_taste < 70'))


# In[14]:


# len(data.query('percentage_recovery_taste > 100'))


# here there are 721 (!) cases

# # DATE DELTA

# In[15]:


data['Date_of_onset'] = pd.to_datetime(data['Date_of_onset'])


# In[16]:


data['days_from_onset'] = ((data['Smell_after_illness_Time_Stamp'] - data['Date_of_onset']).dt.total_seconds()/(60*60*24))


# In[17]:


data['days_from_onset'].head()


# In[25]:


X = data[['days_from_onset']]
y = data['Smell_before_illness'] - data['Smell_after_illness']


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[29]:


import matplotlib.pyplot as plt
print('Coefficients: \n', lm.coef_)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[31]:


from sklearn import metrics
import numpy as np

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




