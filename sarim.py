import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from pmdarima.arima import auto_arima
def test_stationarity(timeseries):

    #Determing rolling statistics
    
    rolmean = timeseries.rolling( window=30).mean()
    rolstd = timeseries.rolling( window=30).std()

    #Plot rolling statistics:
   
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries['ZDAY_ST_QTY'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput) 




data=pd.read_csv("data_sales.csv")

'''
data1=pd.read_csv("data_sales.csv")
del data1['ZDAY_ST_QTY']
del data['ZDAY_ST_VAL']
'''
data['Cal_Date']=pd.to_datetime(data['Cal_Date'], format='%d-%m-%Y')
data.set_index(['Cal_Date'], inplace=True)
#test_stationarity(data)
train = data[:int(0.7*(len(data)))]
valid = data[int(0.7*(len(data))):]
data_log=np.square(train)

#moving_avg=data_log.rolling(window=30).mean()
#data_r=data_log-moving_avg
#data_r.dropna(inplace=True)
data_diff=data_log - data_log.shift()
data_diff.dropna(inplace=True)
#test_stationarity(data_diff)
'''
decomposition=seasonal_decompose(data_log)
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid
plt.subplot(411)
plt.plot(data_log,label='Original')
plt.legend(loc="best")
plt.subplot(412)
plt.plot(trend,label='trend')
plt.legend(loc="best")
plt.subplot(413)
plt.plot(seasonal,label='seasonal')
plt.legend(loc="best")
plt.subplot(414)
plt.plot(residual,label='residual')
plt.legend(loc="best")
plt.tight_layout()

decomposed_squared_data=residual
decomposed_squared_data.dropna(inplace=True)
#test_stationarity(decomposed_squared_data)
'''
'''
lag_acf=acf(data,nlags=61)
lag_pacf=pacf(data,nlags=61,method='ols')
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),color='gray')
plt.title('Auto Correlation')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data)),color='gray')
plt.axhline(y=1.96/np.sqrt(len(data)),color='gray')
plt.title("Paritial Auto Correlation")
plt.tight_layout()
'''
'''
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data_diff['ZDAY_ST_QTY'].diff().dropna(), lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data_diff['ZDAY_ST_QTY'].diff().dropna(), lags=50, ax=ax2)
#plt.show()
'''
#print(type(data_diff))

#
'''
mod = sm.tsa.statespace.SARIMAX(data_diff, trend='n', order=(1,1,1), seasonal_order=(2,1,1,7))
results = mod.fit()
print (results.summary())
#del data1['ZDAY_ST_QTY']


mod1 = sm.tsa.statespace.SARIMAX(data1, trend='n', order=(2,1,1), seasonal_order=(0,1,1,7))
results1 = mod1.fit()
print (results1.summary())



data_diff['forecast'] = results.predict(start = 41, end= 61, dynamic= True)  

data_diff[['ZDAY_ST_QTY', 'forecast']].plot(figsize=(12, 8)) 
plt.plot(bbox_inches='tight')
plt.show()


start = datetime.datetime.strptime("2017-08-01", "%Y-%m-%d")
date_list = [start + relativedelta(day=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= data_diff.columns)
data_diff = pd.concat([data_diff, future])
data1=pd.concat([data1, future])
data_diff['forecast'] = results.predict(start = 61, end = 72, dynamic= True)  
data_diff[['ZDAY_ST_QTY', 'forecast']].plot(figsize=(12, 8)) 
plt.plot(bbox_inches='tight')
plt.show()
data1['forecast'] = results.predict(start = 61, end = 72, dynamic= True)  
data1[['ZDAY_ST_VAL', 'forecast']].plot(figsize=(12, 8)) 
plt.plot(bbox_inches='tight')
plt.show()
'''
#training model
model = auto_arima(data_diff, trace=True,start_p=0, start_q=0, start_P=0, start_Q=0,
                  max_p=10, max_q=10, max_P=10, max_Q=10, seasonal=True,
                  stepwise=False, suppress_warnings=True, D=1, max_D=10,
                  error_action='ignore',approximation = True)
#fitting model
model.fit(data_diff)

y_pred = model.predict(n_periods=len(valid))
from sklearn.metrics import r2_score
acc = r2_score(valid.values, y_pred)
print(acc)
