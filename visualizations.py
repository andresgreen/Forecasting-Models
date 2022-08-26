# Import libraries
#from datetime import datetime
#import numpy as np
#import pandas as pd
#import matplotlib.pylab as plt
#from matplotlib.pylab import rcParams
# from prophet import Prophet
# from prophet.plot import plot_plotly, plot_components_plotly
# from prophet.diagnostics import cross_validation
# from prophet.diagnostics import performance_metrics
# from prophet.plot import plot_cross_validation_metric

def prophet_viz(m):
    future = m.make_future_dataframe(periods=12, freq='m')
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
    return fig1, fig2