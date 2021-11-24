#시계열 분석 : 현시점까지의 데이터로 앞으로 어떤 패턴의 차트를 그릴지 예측하는 분석 기법
#비트 코인 차트 데이터를 예시로 데이터 분석 진행

#import
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as tsa


source = 'market-price.csv'
bitcoin_df = pd.read_csv(source,names=['day','price'])

#print(bitcoin_df)

#set day column as index
bitcoin_df.index = bitcoin_df['day']
bitcoin_df.set_index('day',inplace=True)

#bitcoin_df.plot()
#plt.show()

#ARIMA 
model = sm.tsa.ARIMA(bitcoin_df.price.values,order=(2,1,2)) #ar =2 , differencing = 1, ma=2

model_fit = model.fit()
print(model_fit.summary())

#ARIMA 모델 학습에서의 예측 결과
forecast_data = model_fit.forecast(step=5) # 학습 데이터셋으로부터 5일 뒤의 결과 예측

#test_df = 'market-price-test.csv'
#bitcoin_test_df = pd.read_csv(test_df,names=['ds','y'])

pred_y = forecast_data[0].tolist()
pred_y_lower = []
pred_y_upper = []

#for lower_upper in forecast_data[2]:
#    lower = lower_upper[0]
#    upper = lower_upper[1]
#    pred_y_lower.append(lower)
#    pred_y_upper.append(upper)
plt.plot(pred_y,color="gold")
#plt.plot(pred_y_lower,color="red")
#plt.plot(pred_y_upper,color="blue")
plt.show()
