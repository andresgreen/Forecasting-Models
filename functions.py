import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric


# Function that will unpivot tables with dates as columns, and return an unpivoted dataframe with a date column and a value column within the selected periods. 
def brand_df(data,brand_name,start_date,end_date):
    df = data.loc[data['Brand'] == brand_name]
    id_vars = list(df.columns)[0] # Isolate ID
    value_vars = list(df.drop(id_vars,axis=1)) # Isolate Value Variables
    brand = pd.melt(df, id_vars=id_vars,value_vars=value_vars,value_name='Sales',var_name='Date')
    brand['Sales'] = brand['Sales'] #.astype(int)
    brand = brand.loc[(brand['Date']>start_date) & (brand['Date']<end_date) ]
    brand['Date'] = pd.to_datetime(brand['Date'])
    brand = brand.set_index('Date')
    brand.drop(columns='Brand',inplace=True)
    return brand

# Simple EDA
def data_desc(brand):
    return brand.head(),print("Empty values "+str(brand.isna().sum())),brand.plot()

# Facebook's prophet Model

def prophet_mod(brand):
    brand = brand.reset_index()
    brand.rename(columns = {'Date':'ds', 'Sales':'y'}, inplace = True)
    m = Prophet()
    m = m.fit(brand)
    df_cv = cross_validation(m, initial='730 days', period='365 days', horizon = '365 days')
    df_p = performance_metrics(df_cv)
    return df_cv,df_p