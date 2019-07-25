# -*- coding: utf-8 -*-
## Load Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
import scipy.stats as stats
import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from collections import Counter
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
import itertools
plt.rcParams['figure.dpi'] = 200
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from IPython.display import HTML, Math
display(HTML("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/"
             "latest.js?config=default'></script>"))
Math(r"e^\alpha")

import unicodedata

def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)
    
    
    
    
## Load Data

air_reserve_d=pd.read_csv('/content/drive/My Drive/air_reserve.csv')
hpg_reserve_d=pd.read_csv('/content/drive/My Drive/hpg_reserve.csv')
air_store_info_d=pd.read_csv('/content/drive/My Drive/air_store_info.csv')
air_visit_d=pd.read_csv('/content/drive/My Drive/air_visit_data.csv')
date_info_d=pd.read_csv('/content/drive/My Drive/date_info.csv')
store_id_relation_d=pd.read_csv('/content/drive/My Drive/store_id_relation.csv')

## Transformations of data


# (a) Strip accents of air_area_name in air_store_info

air_store_info_d["air_area_name"]=air_store_info_d["air_area_name"].apply(strip_accents)

# (b) Add datetime to date attributes
air_reserve_d[["visit_datetime", "reserve_datetime"]]=air_reserve_d[["visit_datetime", "reserve_datetime"]].apply(pd.to_datetime)
hpg_reserve_d[["visit_datetime", "reserve_datetime"]]=hpg_reserve_d[["visit_datetime", "reserve_datetime"]].apply(pd.to_datetime)
air_visit_d["visit_date"]=air_visit_d["visit_date"].apply(pd.to_datetime)
date_info_d["calendar_date"]=date_info_d["calendar_date"].apply(pd.to_datetime)

## Initial forecasting for a particular store


def construct_daily_time_series(time_series, min_date=None, max_date=None):
  """
  """
  time_series_dailyvisitors_d=time_series[["visitors"]].groupby(pd.TimeGrouper(freq='d')).mean().reset_index()
  time_series_dailyvisitors_d["visit_date"]=time_series_dailyvisitors_d["visit_date"].dt.date
  
  if min_date==None:
    min_date=time_series_dailyvisitors_d["visit_date"].min()
    max_date=time_series_dailyvisitors_d["visit_date"].max()
    
  date_range=pd.date_range(min_date, max_date)
  
  date_range_d=pd.DataFrame({'visit_date':date_range})
  date_range_d["day_of_week"]=date_range_d["visit_date"].dt.dayofweek
  date_range_d["visit_date"]=date_range_d["visit_date"].dt.date
  
  dailyvisitors_d=date_range_d.merge(time_series_dailyvisitors_d,
                                    on="visit_date",
                                    how="left")
  
  # Fill Na by day of the week median
  dailyvisitors_d["visitors"]=dailyvisitors_d["visitors"].fillna(dailyvisitors_d.groupby("day_of_week")["visitors"].transform('median'))
  

  return dailyvisitors_d
## Analyze results from ARIMA analysis

results_ARIMA=pd.read_csv('results_measures_ARIMA_new.csv')

genre_names=list(set(results_ARIMA["air_genre_name"]))

print results_ARIMA.iloc[0]


p_opt_MAE=[]
q_opt_MAE=[]
p_opt_RMSE=[]
q_opt_RMSE=[]
MAE_opt=[]
RMSE_opt=[]

for i in range(len(genre_names)):
  
  temp_results=results_ARIMA.query("air_genre_name=='%s'"%genre_names[i])
  test_MAE_opt=temp_results.sort_values(by="test_MAE", ascending=True).iloc[0][["test_MAE", "p_values", "q_values"]]
  test_RMSE_opt=temp_results.sort_values(by="test_RMSE", ascending=True).iloc[0][["test_RMSE", "p_values", "q_values"]]
  
  p_opt_MAE.append(test_MAE_opt["p_values"])
  q_opt_MAE.append(test_MAE_opt["q_values"])
  MAE_opt.append(test_MAE_opt["test_MAE"])
  p_opt_RMSE.append(test_RMSE_opt["p_values"])
  q_opt_RMSE.append(test_RMSE_opt["q_values"])
  RMSE_opt.append(test_RMSE_opt["test_RMSE"])
  
  
print p_opt_MAE
  
# Create table for test_MAE

genre_names[0]="Okonomiyaki/..."
data_MAE=np.stack((genre_names, p_opt_MAE, q_opt_MAE, np.round(MAE_opt, 2))).T
print table_MAE

fig, ax = plt.subplots()
ax.axis("off")

table_MAE=ax.table(cellText=data_MAE, bbox=[0, 0, 1.5, 1.3],
                  colLabels=["Genre Name", "p", "q", "MAE"])

table_MAE.auto_set_font_size(False)
table_MAE.set_fontsize(10)

# Create table for test_RMSE

data_RMSE=np.stack((genre_names, p_opt_RMSE, q_opt_RMSE, np.round(RMSE_opt, 2))).T


table_RMSE=ax.table(cellText=data_RMSE, bbox=[1.7, 0, 1.5, 1.3],
                   colLabels=["Genre Name", "p", "q", "RMSE"])
table_RMSE.auto_set_font_size(False)
table_RMSE.set_fontsize(10)

fig.set_facecolor("navajowhite")

title_MAE="Optimal Parameters for Different Genres - MAE"
title_RMSE="Optimal Parameters for Different Genres - RMSE"

title_alg="ARIMA(p,d,q) Model - with statsmodels package (Python)"
com_alg="Based on utilizing d=7, i.e. a differencing of lag 7 \n" \
"To find optimal p and q, two metrics, MAE and RMSE, are evaluated on \n" \
"on a test set consisting of a week, given an ARIMA model trained \n" \
"on a training set. \n" \
"For the search of optimal p, q, all ARIMA models for p=1,2,...,7 \n" \
"and q=1,2,...,7 have been considered"
com_table="In above tables, the optimal choices of p and q for \n" \
"each genre is presented. \n" \
"MAE is the mean of the residuals, with respect to true and predicted \n" \
"time series values. \n" \
"RMSE is the square root of the mean of the residuals squared"

fig.text(0.4, 1.13, title_MAE, fontsize=13, fontweight="bold")
fig.text(1.6,1.13, title_RMSE, fontsize=13, fontweight="bold")
fig.text(0.3, 0.05, title_alg, fontsize=13, fontweight="bold")
fig.text(0.3, -0.27, com_alg, fontsize=13)
fig.text(1.5, -0.2,  com_table, fontsize=13)
 

  
  
