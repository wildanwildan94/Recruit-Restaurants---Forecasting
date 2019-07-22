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
## Find the optimal ARIMA model with a 7day differencing
## on a prediction on 7 days ahead, by iterating 
## over different models

visit_genre_d=air_visit_d.merge(air_store_info_d[["air_store_id", "air_genre_name"]], on="air_store_id",
                               how="left")


visit_genre_d=visit_genre_d.set_index("visit_date")


genres_name=list(set(air_store_info_d["air_genre_name"]))


p_values=np.arange(1,8)
q_values=np.arange(1,4)

genre_name_array=[]
p_values_array=[]
q_values_array=[]
train_MAE_array=[]
train_RMSE_array=[]
test_MAE_array=[]
test_RMSE_array=[]

for i in range(len(genres_name)):

  # Define the current genre considered
  temp_genre_name=genres_name[i]
  
  print "---"
  print "Currently computing for genre: %s"%temp_genre_name
  
  # Consider all data values associated with genre
  visit_tempgenre_d=visit_genre_d.query("air_genre_name=='%s'"%temp_genre_name)
  
  # Construct daily visitor count for genre
  dailyvisit_tempgenre_d=construct_daily_time_series(visit_tempgenre_d)
  
  # Transform datatype to datetime
  dailyvisit_tempgenre_d["visit_date"]=pd.to_datetime(dailyvisit_tempgenre_d["visit_date"])
  
  # Define dates for the training and test data
  last_day_data=dailyvisit_tempgenre_d["visit_date"].max()
  train_upper_day=(last_day_data-datetime.timedelta(days=14)).strftime('%Y-%m-%d')
  test_upper_day=(last_day_data-datetime.timedelta(days=7)).strftime('%Y-%m-%d')
  
  train_dailyvisit_genre_d=dailyvisit_tempgenre_d.query("visit_date<='%s'"%train_upper_day).sort_values(by="visit_date",
                                                                                                   ascending=True)
  
  test_dailyvisit_genre_d=dailyvisit_tempgenre_d.query("visit_date<='%s' and visit_date>'%s'"%(test_upper_day,
                                                                                          train_upper_day)).sort_values(by="visit_date",
                                                                                                                       ascending=True)
  
  
  
  train_dailyvisit_genre_d["visitors_7day_diff"]=train_dailyvisit_genre_d["visitors"].diff(periods=7)
  train_dailyvisit_genre_d["visitors_7day_shift"]=train_dailyvisit_genre_d["visitors"].shift(periods=7)
  
  train_dailyvisit_genre_d=train_dailyvisit_genre_d.dropna(subset=["visitors_7day_diff",
                                                                  "visitors_7day_shift"])
  
  # Iterate over different ARIMA orders, p and q
  for p in p_values:
    for q in q_values:
      model=ARIMA(train_dailyvisit_genre_d["visitors_7day_diff"], order=(p,0,q),
                 dates=train_dailyvisit_genre_d["visit_date"],
                 freq="D")
      
      fit=model.fit(disp=0)


      # Add predictions
      train_dailyvisit_genre_d["visitors_7day_diff_pred"]=fit.predict(train_dailyvisit_genre_d["visit_date"].min(),
                                                                   train_dailyvisit_genre_d["visit_date"].max()).values
      
      
      test_dailyvisit_genre_d["visitors_7day_diff"]=test_dailyvisit_genre_d["visitors"].values-train_dailyvisit_genre_d.tail(7)["visitors"].values
      test_dailyvisit_genre_d["visitors_7day_shift"]=train_dailyvisit_genre_d.tail(7)["visitors"].values
      test_dailyvisit_genre_d["visitors_7day_diff_pred"]=fit.predict(test_dailyvisit_genre_d["visit_date"].min(),
                                                              test_dailyvisit_genre_d["visit_date"].max()).values

      # Compute MAE and RMSE of train & test

      train_MAE=np.mean(np.abs(train_dailyvisit_genre_d["visitors_7day_diff"].values-train_dailyvisit_genre_d["visitors_7day_diff_pred"].values))
      test_MAE=np.mean(np.abs(test_dailyvisit_genre_d["visitors_7day_diff"].values-test_dailyvisit_genre_d["visitors_7day_diff_pred"].values))

      train_RMSE=np.sqrt(np.mean((train_dailyvisit_genre_d["visitors_7day_diff"].values-train_dailyvisit_genre_d["visitors_7day_diff_pred"].values)**2))
      test_RMSE=np.sqrt(np.mean((test_dailyvisit_genre_d["visitors_7day_diff"].values-test_dailyvisit_genre_d["visitors_7day_diff_pred"].values)**2))
      
      p_values_array.append(p)
      q_values_array.append(q)
      genre_name_array.append(temp_genre_name)
      
      train_MAE_array.append(train_MAE)
      test_MAE_array.append(test_MAE)
      
      train_RMSE_array.append(train_RMSE)
      test_RMSE_array.append(test_RMSE)
      
      
    


results_measures=pd.DataFrame({'p_values':p_values_array,
                              'q_values':q_values_array,
                              'air_genre_name':genre_name_array,
                              'train_MAE':train_MAE_array,
                              'test_MAE':test_MAE_array,
                              'train_RMSE':train_RMSE_array,
                              'test_RMSE':test_RMSE_array})
results_measures.to_csv('results_measures_ARIMA_new.csv', index=False)
      

  

