# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 21:46:37 2021

@author: pc2
"""
from fredapi import Fred
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import MinMaxScaler

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs

import pandas_datareader as pdr
from datetime import datetime
import pandas_datareader as data 

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
#Datareader API로 Yahoo finance 주식 종목 가져오기

#Henry Hub Spot Price
start=datetime(2012,1,1)
end=datetime(2019,10,31)

fred=Fred(api_key='5b00d2114222da72aab563643d32fb11')
data=fred.get_series('SP500')

HH=fred.get_series('DHHNGSP',start,end)
HH.plot()


#WTI Spot Price
WTI=fred.get_series('DCOILWTICO',start,end)
WTI.plot()


#Brent Spot Price
Brent=fred.get_series('DCOILBRENTEU',start,end)
Brent.plot()

#dubai Spot Price
Dubai=fred.get_series('DCOILBRENTEU',start,end)
Dubai.plot()


#NG 선물
df_NGF=pdr.DataReader('QG=F','yahoo',start,end)['Adj Close']
df_NGF.head()
df_NGF

tmp_HH = HH.copy()
tmp_df_NGF = df_NGF.copy()
tmp_HH.columns = ['Date', 'f_close']
tmp_df_NGF.columns = ['Date', 'a_close']

tmp_merged = pd.merge(tmp_HH, tmp_df_NGF)

data=pd.concat([HH,df_NGF],axis=1,keys=['Spot','Future'])
data=data.dropna(axis=0)
data.head()

#선물 가격과 natural gas spot price VAR 분석
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

data.plot(figsize=(8,5))

#ADF TEST
adfuller_test=adfuller(data['Spot'],autolag="AIC")
print("ADF test statics: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1])) #non-stationary

adfuller_test=adfuller(data['Future'],autolag="AIC")
print("ADF test statics: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1])) #non-stationary, 0.056, stationary에 가깝?

#차분 후 ADF Test
data_diff=data.diff().dropna()

adfuller_test=adfuller(data_diff['Spot'],autolag="AIC")
print("ADF test statics: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1])) #stationary, 1차 차분

adfuller_test=adfuller(data_diff['Future'],autolag="AIC")
print("ADF test statics: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1])) #stationary, 1차 차분

#Training and Test
train=data_diff.iloc[:-10,:]
test=data_diff.iloc[-10:,:]

#VAR AIC Value로 차수 정하기
forecasting_model=VAR(train)
results_aic=[]
for p in range(1,20):
    results=forecasting_model.fit(p)
    results_aic.append(results.aic)

sns.set()
plt.plot(list(np.arange(1,20,1)),results_aic)
plt.xlabel('order')
plt.ylabel('AIC')
plt.show()

results=forecasting_model.fit(2)
results.summary()

#원유와의 Coupling 여부 확인
data=pd.concat([HH,WTI],axis=1,keys=['NG','WTI'])
data=data.dropna(axis=0)
data.head()

#선물 가격과 natural gas spot price VAR 분석
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

data.plot(figsize=(8,5))

#ADF TEST
adfuller_test=adfuller(data['NG'],autolag="AIC")
print("ADF test statics: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1])) #non-stationary

adfuller_test=adfuller(data['WTI'],autolag="AIC")
print("ADF test statics: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1])) #non-stationary, 0.514

#차분 후 ADF Test
data_diff=data.diff().dropna()

adfuller_test=adfuller(data_diff['NG'],autolag="AIC")
print("ADF test statics: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1])) #stationary, 1차 차분

adfuller_test=adfuller(data_diff['WTI'],autolag="AIC")
print("ADF test statics: {}".format(adfuller_test[0]))
print("p-value: {}".format(adfuller_test[1])) #stationary, 1차 차분

#Training and Test
train=data_diff.iloc[:-10,:]
test=data_diff.iloc[-10:,:]

#VAR AIC Value로 차수 정하기
forecasting_model=VAR(train)
results_aic=[]
for p in range(1,20):
    results=forecasting_model.fit(p)
    results_aic.append(results.aic)

sns.set()
plt.plot(list(np.arange(1,20,1)),results_aic)
plt.xlabel('order')
plt.ylabel('AIC')
plt.show()

results=forecasting_model.fit(5)
results.summary()

#text mining
import urllib3
from bs4 import BeautifulSoup
import re
