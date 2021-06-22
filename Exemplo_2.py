# carregar um arquivo do dico rígido
# from builtins import print

import pandas as pd
import plotly.express as px

data = pd.read_csv('datasets/kc_house_data.csv')

# # # EXERCICIO 1
data['house_age'] = None
data.loc[data.date >= '2014-01-01', 'house_age'] = 'new house'
data.loc[data.date < '2014-01-01', 'house_age'] = 'old house'
print(data.columns)
print(data['house_age'].head())
#
# #EXERCICIO 2
data['dormitory_type'] = None
data.loc[data.bedrooms == 1, 'dormitory_type'] = 'studio'
data.loc[data.bedrooms == 2, 'dormitory_type'] = 'apartment'
data.loc[data.bedrooms > 2, 'dormitory_type'] = 'house'
print(data['dormitory_type'].head(351))
#
# # #EXERCICIO 3
data['condition_type'] = None
data.loc[data.condition <= 2, 'condition_type'] = 'bad'
data.loc[(data.condition == 3) | (data.condition == 4), 'condition_type'] = 'regular'
# # data.loc[data.condition == 4, 'condition_type'] = 'regular'
data.loc[data.condition >= 5, 'condition_type'] = 'good'
print(data['condition_type'].head(39))
# #
# # # EXERCICIO 4
# print(data.dtypes)
# data['condition'] = data['condition'].astype( str )
# print(data.dtypes)
# # #
# # # #EXERCICIO 5
# print(data.columns)
# data = data.drop( ['sqft_living15'], axis=1 )
# data = data.drop( ['sqft_lot15'], axis=1 )
# print(data.columns)
# # #
# # # #EXERCICIO 6
# data['yr_built'] = pd.to_datetime(data['yr_built'])
# print(data.dtypes)
# # #
# # # #EXERCICIO 7
# data['yr_renovated'] = pd.to_datetime(data['yr_renovated'])
# print(data.dtypes)
# # #
# # # # EXERCICIO 8
# print(data.sort_values('yr_built', ascending=True)['yr_built'])
# #
# # # # EXERCICIO 9
# data['yr_renovated'] = pd.to_datetime(data['yr_renovated'], errors="coerce", format= "%Y")
# print(data.sort_values(['yr_renovated'], ascending=True) ['yr_renovated'])
# #
# # # EXERCICIO 10
print(data[data['floors'] == 2]['floors'])
# #
# # # EXERCICIO 11
print(data[data['condition_type'] == 'regular']['condition_type'])
#
# #EXERCICIO 12
print(data[(data['condition_type'] == 'bad') & (data['waterfront'] > 0)]['condition_type'])
#
# #EXERCICIO 13
print(data[(data['condition_type'] == 'good') & (data['house_age'] == 'new house')]['condition_type'])

# EXERCICIO 14
maior = data.loc[data['dormitory_type'] == 'studio', 'price'].max()
print(maior)

# EXERCICIO 15
print(data[(data['yr_renovated'] == 2015) & (data['dormitory_type'] == 'apartment')]['id'])

# EXERCICIO 16
print(data.loc[data['dormitory_type'] == 'house', 'floors'].max())

# EXERCICIO 17
print(data[(data['yr_renovated'] == 2014) & (data['house_age'] == 'new house')]['id'])

# EXERCICIO 18
# 18.1
print(data[['id', 'date', 'price', 'floors', 'zipcode']])
# 18,2
print(data.iloc[0:20, 0:5])
# 18.3
print(data.loc[0:7, 'grade'])
# 18.4
cols = [False, False, True, True, False, False, False, True, True, False, True, False, False, False, False, False, True, False, False, False, False, False, False, False]
print(data.loc[0:15, cols])

# EXERCICIO 19
report = data[['id', 'price', 'condition_type', 'floors']]
report.to_csv('datasets/report_aula02.csv', index = False)

# EXERCICIO 20

data_mapa = data[['floors', 'lat', 'long', 'price']]
mapa = px.scatter_mapbox(data_mapa, lat='lat', lon='long',
                          hover_name='floors',
                          hover_data=['price'],
                          color_discrete_sequence=['green'],
                          zoom=3,
                          height=300)
mapa.update_layout(mapbox_style='open-street-map')
mapa.update_layout(height=600, margin={'r':0, 't':0, 'l':0, 'b':0})
mapa.show()