# Import modules 
import functions as func
import data as d
import visualizations as viz

# Import data

data = d.brand_data

start_date='2017-06-01'

end_date='2022-08-1'

brand = func.brand_df(data,'TAJIN',start_date,end_date)

print(brand.head())

tajin_fc = func.prophet_mod(brand)

print(tajin_fc)