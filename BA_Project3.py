import pandas as pd

df = pd.read_pickle( 'Project-3_NYC_311_Calls.pkl')

df = df.set_index(pd.DatetimeIndex(df['Created Date']))
del df['Created Date']

df.head()

#1
daily_complaints = df['Unique Key'].resample('D').count()
daily_complaints

daily_complaints_2022 = daily_complaints['2022']
daily_complaints_2022

average_daily_complaints_2022 = daily_complaints_2022.mean()
average_daily_complaints_2022

#2
daily_complaints.idxmax()

#3
date_to_filter = '2020-08-04'
date_to_filter = pd.Timestamp(date_to_filter)

temp_df = df[df.index.date == date_to_filter.date()]
temp_df.head()

temp_df['Complaint Type'].value_counts()

#4
monthly_complaints = df['Unique Key'].resample('M').count()
quietest_month = monthly_complaints.idxmin()
quietest_month

#5
from statsmodels.tsa.seasonal import seasonal_decompose

daily_complaints = df['Unique Key'].resample('D').count()

result = seasonal_decompose(daily_complaints, model='add', period=365)

seasonal_component = result.seasonal

round(seasonal_component['2020-12-25'])

#6
daily_complaints.autocorr(lag=1)

!pip install prophet

#7
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt
data = daily_complaints.reset_index()
data.columns = ['ds','y']
train = data.iloc[:-90]
test = data.iloc[-90:]

model = Prophet()
model.fit(train)

future = model.make_future_dataframe(periods=90)

forecast = model.predict(future)

y_true = test['y'].values

y_pred = forecast.iloc[-90:]['yhat'].values

rmse = sqrt(mean_squared_error(y_true, y_pred))
rmse










