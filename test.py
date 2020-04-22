import pandas as pd
#import plotly.plotly as ply
import matplotlib.pyplot as plt
import cufflinks as cf
from pyramid.arima import auto_arima

data = pd.read_csv('d.csv',index_col=1,low_memory=False)

print(type(data))
print(len(data))
print(data)
columnsNamesArr = data.columns.values
print(columnsNamesArr)
data=data.drop(columns=['Cust_no'])
data1=data.drop(columns=['Qty'])
data=data.drop(columns=['Value'])
print(data1)
print(data)
data.index = pd.to_datetime(data.index)
# plotting the points  
plt.plot(data.index,data['Qty']) 
  
# naming the x axis 
plt.xlabel('x - axis') 
# naming the y axis 
plt.ylabel('y - axis') 
  
# giving a title to my graph 
plt.title('My first graph!') 
  
# function to show the plot 
plt.draw()
#data=data.drop(columns=['Cust_no'])
print(data)

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

train = data.loc['01-04-2018':'01-12-2018']
test = data.loc['01-12-2018':]
stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=12)
print(future_forecast)