from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
#import plotly.plotly as ply
import matplotlib.pyplot as plt
import cufflinks as cf
#from pyramid.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import random
def home(request):
	return render(request,'stats/home.html')

def predict(request):

	cust_id=request.POST['cust_no']
	#print(type(cust_id))
	Start = request.POST['st_time']
	End = request.POST['end_time']
	s_m=int(''.join(Start[6:]))
	e_m=int(''.join(End[6:]))
	s_y=int(''.join(Start[:4]))
	e_y=int(''.join(End[:4]))
	print(s_m,e_m,s_y,e_y)
	period=(e_y-s_y)*12+(e_m-s_m+1)
	#print(period)
	data = pd.read_csv('d2.csv')

	data=data[data.Cust_no==float(cust_id)]
	
	data['datetime'] = pd.to_datetime(data['month'])
	data = data.set_index('datetime')
	data.drop(['month'], axis=1, inplace=True)
	print(data)		
	#data.index = pd.to_datetime(data.index)
	data=data.drop(columns=['Cust_no'])
	data1=data.drop(columns=['Qty'])
	data=data.drop(columns=['Value'])
	
		#print(data1)
	#print("01-"+s_m+"-"+s_y)
	#print(data)
	data.index = pd.to_datetime(data.index)
	#----------------
	'''
	decomposition=seasonal_decompose(data,freq=6)
	trend=decomposition.trend
	seasonal=decomposition.seasonal
	residual=decomposition.resid
	plt.subplot(411)
	plt.plot(data,label='Original')
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
	#plt.show()	
	decomposed_squared_data=residual
	decomposed_squared_data.dropna(inplace=True)
	
	print("rgtvrv")
	'''
	#test_stationarity(decomposed_squared_data)
	stepwise_model = auto_arima(data, start_p=0, start_q=0,
	                           max_p=3, max_q=3	, m=1,
	                           start_P=0, seasonal=True,
	                           d=1, D=1, trace=True,
	                           error_action='ignore',  
	                           suppress_warnings=True, 
	                           stepwise=True)
	print(stepwise_model.aic())
	
	train = data.loc['01-04-2018':'01-12-2018']
	stepwise_model.fit(train)

	future_forecast = stepwise_model.predict(n_periods=period)
	#return HtttpResponse(future_forecast)
	context={'future_forecast':future_forecast}
	return render(request,'stats/predict.html',context)
	

