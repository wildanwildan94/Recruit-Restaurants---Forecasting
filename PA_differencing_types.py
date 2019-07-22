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
  
  
  
## Analyze different differencing of time series


air_visit_store_d=air_visit_d.query("air_store_id=='air_ba937bf13d40fb24'")

air_visit_store_d=air_visit_d.query("air_store_id=='air_ba937bf13d40fb24'")
air_visit_store_d=air_visit_d.set_index("visit_date")
dailyvisitors_store_d=construct_daily_time_series(air_visit_store_d)



"""

dailyvisitors_store_d["log_visitors"]=dailyvisitors_store_d["visitors"].apply(np.log)


dailyvisitors_store_d.sort_values(by="visit_date", ascending=True, inplace=True)

dailyvisitors_store_d["vis_7day"]=dailyvisitors_store_d["visitors"].diff(periods=7)
dailyvisitors_store_d["logvis_7day"]=dailyvisitors_store_d["log_visitors"].diff(periods=7)
dailyvisitors_store_d["vis_7day"]=dailyvisitors_store_d["vis_7day"].fillna(np.mean(dailyvisitors_store_d["vis_7day"]))
dailyvisitors_store_d["logvis_7day"]=dailyvisitors_store_d["logvis_7day"].fillna(np.mean(dailyvisitors_store_d["logvis_7day"]))


# (a) Compute a 7 day differencing


# (b) Viusalize the 7 day differencing

fig, ax = plt.subplots()


ax.plot(dailyvisitors_store_d.index,
       dailyvisitors_store_d["vis_7day"])

fig, ax = plt.subplots()

ax.plot(dailyvisitors_store_d.index,
       dailyvisitors_store_d["logvis_7day"])
"""
# (c) Want to test different differening orders

diff_array=[1,3,5,7]
dates=dailyvisitors_store_d["visit_date"]

dailyvisitors_store_d.sort_values(by="visit_date", ascending=True, inplace=True)

fig, ax = plt.subplots(2,4)
for i in range(len(diff_array)):
  diff_ts=dailyvisitors_store_d["visitors"].diff(periods=diff_array[i])
  diff_ts=diff_ts.fillna(np.mean(diff_ts))

  
  ax[0, i].plot(dates, diff_ts, color="crimson")
  ax[0, i].set_title("Differencing of Lag: %s"%diff_array[i])
  
  ax[0,i].tick_params(axis="x", rotation=40)
  ax[0,i].set_facecolor("navajowhite")
  
  ax[1,i].acorr(diff_ts, maxlags=20)
  ax[1,i].set_title("Autocorrelation Function")
  ax[1,i].set_facecolor("navajowhite")

  
ax[1,0].set_xlabel("Lag")
ax[1,0].set_ylabel("Autocorrelation")
ax[0,0].set_ylabel("Differenced Visitors")
fig.subplots_adjust(bottom=-0.7, right=2, hspace=0.3)
fig.suptitle("Differenced Time Series of Visitors for store: %s \n And Autocorrelation Function"%'air_ba937bf13d40fb24',
            x=1.1, y=1.03)

fig.set_facecolor("floralwhite")

com_res="\n".join((r"$\cdot$ " "To analyze the potential \n" \
                  "of modeling a stationary time-series,  \n" \
                  "the visitors over time have been \n" \
                   "differenced for different lags",
                  r"$\cdot$ " "As indicated, there is \n" \
                  "a prevalent correlation for most lags \n" \
                  "for differencing of order 1,3,5 - however, \n" \
                   "the differenced series of lag 7 has a  \n" \
                  "quickly decreasing and generally low  \n" \
                   "autocorrelation function.",
                  r"$\cdot$ " "The 7-lag differenced \n" \
                  "series implies that a 7-day differencing \n" \
                  "may be an appropriate technique to reach \n" \
                   "stationarity of a time-series",
                  r"$\cdot$ " "That a 7-day differencing works \n" \
                  "fine is not a surprise - the week consists \n" \
                  "of 7 days, so the series essentially reduces \n" \
                  "over-week variation among time-series values"))

box=dict(boxstyle="round", edgecolor="black",
        facecolor="wheat")
fig.text(2.04, -0.1, com_res, fontsize=13,
        bbox=box)
  
  
  
