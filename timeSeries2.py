import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

import os
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

weather_station_location = pd.read_csv("Weather Station Locations.csv")
weather = pd.read_csv("Summary of Weather.csv")

locations = weather_station_location[['WBAN','NAME','STATE/COUNTRY ID','Latitude','Longitude']]
weather = weather[['STA','Date','MeanTemp']]

#location : BINDUKURI
weather_station_id = locations[locations['NAME']=="BINDUKURI"].WBAN
weather_bin = weather[weather['STA'] == int(weather_station_id)]
weather_bin['Date'] = pd.to_datetime(weather_bin['Date'])

#print(weather_bin.shape) #751 rows X 3 cols 

#time series graph 
#plt.figure(figsize=(22,8))
#plt.plot(weather_bin['Date'],weather_bin['MeanTemp'])
#plt.title("Mean Temperature of Bindukuri Area")
#plt.xlabel('Date')
#plt.ylabel('Mean Temperature')
#plt.show()

#create time series from weather
timeSeries = weather_bin[['Date','MeanTemp']]
timeSeries.index = timeSeries['Date']
ts = timeSeries.drop("Date",axis=1)
#print(ts.head(3))
#print(ts.shape)

#decomposing using seasonal_decompose()
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(ts['MeanTemp'],model='additive',freq=7)

#freq : 계절성 주기를 기반으로 설정
#분기별 데이터는 4, 월별 데이터는 12, 일별 데이터는 7로 초기 설정하면서 확인

fig = plt.figure()
fig = result.plot()
fig.set_size_inches(20,15)
#fig.savefig('test.png')

#정상성 확인 (ACF 그래프 그리기)
import statsmodels.api as sm

#fig = plt.figure(figsize=(20,8))
#ax1 = fig.add_subplot(211)
#fig = sm.graphics.tsa.plot_acf(ts,lags=20,ax=ax1)
#fig.savefig('acf.png')

#Augmented Dickey-Fuller test 
#귀무 가설 : 자료에 단위근이 존재한다 (정상성을 만족하지 못함)
#대립 가설 : 자료가 정상성을 만족한다.
from statsmodels.tsa.stattools import adfuller
result = adfuller(ts)
print('ADF Statistic: %f '% result[0])
print('p-value: %f'% result[1])
print('Critical Values: ')
for key,value in result[4].items():
    print('\t%s : %.3f'%(key,value))

#p-value 가 0.05 를 넘으므로 귀무 가설을 기각하지 못함, 즉, 해당 데이터는 정상성을 만족하지 못함

#1차 차분
ts_diff = ts - ts.shift()
#plt.figure(figsize=(22,8))
#plt.plot(ts_diff)
#plt.title("Differencing method")
#plt.xlabel('Date')
#plt.ylabel('Differencing Mean Temperature')
#plt.show()

#adf 검정 결과 
result = adfuller(ts_diff[1:])
print('ADF Statistic: %f '% result[0])
print('p-value: %f'% result[1])
print('Critical Values: ')
for key,value in result[4].items():
    print('\t%s : %.3f'%(key,value))

# p-value 가 0.05 보다 작으므로 귀무가설을 기각(정상성을 가지게 됨)

