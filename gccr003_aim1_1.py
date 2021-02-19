# -*- coding: utf-8 -*-
"""
Created on 

@author: asus
"""

import pandas as pd
import datetime as dt
gccrdata=pd.read_excel('data.xlsx')
data=gccrdata.to_csv('GCCRData03.csv')
data=pd.read_csv('GCCRData03')
data.head()
data.info()
data.describe()
data['percentage_recovery_smell'] = data['Smell_current'] / data['Smell_before_illness'] * 100
data['percentage_recovery_taste'] = data['Taste_current'] / data['Taste_before_illness'] * 100
data['percentage_recovery_taste'].head(10)
sum(data['percentage_recovery_smell']<80)
data[data['percentage_recovery_taste']<80].count()[0]
data['Date_of_onset'] = pd.to_datetime(data['Date_of_onset'])
data['Email_Time_Stamp_y'] = pd.to_datetime(data['Email_Time_Stamp_y'])
data['days_from_onset'] = ((data['Email_Time_Stamp_y'] - data['Date_of_onset']).dt.total_seconds()/(60*60*24))
data['days_from_onset'].head(10)

# https://www.analyticsvidhya.com/blog/2020/12/a-brief-introduction-to-survival-analysis-and-kaplan-meier-estimator/
