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


## 7-day differencing of series of different genres


visit_genre_d=air_visit_d.merge(air_store_info_d[["air_store_id", "air_genre_name"]], on="air_store_id",
                               how="left")



print visit_genre_d.iloc[0]


# def construct_daily_time_series(time_series, min_date=None, max_date=None):

# (a) Visualize the 7-day difference time-series of each genre
genre_names=list(set(air_store_info_d["air_genre_name"]))

fig, ax = plt.subplots(4,4)

for temp_genre, axes in zip(genre_names, ax.flatten()):
  
  genre_data_d=visit_genre_d.query("air_genre_name=='%s'"%temp_genre).sort_values(by="visit_date")
  genre_data_d=genre_data_d.set_index("visit_date")
  
  temp_constr_d=construct_daily_time_series(genre_data_d)
  

  temp_diff_ts=temp_constr_d["visitors"].diff(periods=7)
  temp_diff_ts=temp_diff_ts.fillna(np.mean(temp_diff_ts))
  
  axes.plot(temp_constr_d["visit_date"],
         temp_diff_ts, color="crimson")
  axes.set_title("%s"%temp_genre)
  axes.tick_params(axis="x", rotation=30)
  axes.set_facecolor("navajowhite")

 
fig.subplots_adjust(right=2.3, bottom=-1.3, hspace=0.6)
fig.suptitle("Differenced Time Series of Lag: 7 - For Different Genres - 1 Day Average of Visitors", x=1)
ax[3,2].axis("off")
ax[3,3].axis("off")

ax[3,0].set_xlabel("Date", fontsize=14)
ax[3,0].set_ylabel("Visitors", fontsize=14)

fig.set_facecolor("floralwhite")

com_res="\n".join((r"$\cdot$ " "Most differenced series, for different genres, seems probable to be \n" \
                  "stationary - which indicates that a differencing of lag 7 might be appropriate to \n" \
                  "remove seasonality and trend of data.",
                  r"$\cdot$ " "There are still some minor problems, like the extreme values in e.g Creative \n" \
                  "Cuisine or Japanese food - but they occur infrequently, making it a minor problem",
                  r"$\cdot$ " "One could argue that there is still some small trend over time, but \n" \
                  "it seems to appear arbitrarily and with little effect",
                  r"$\cdot$ " "A conclusion is that a differencing of lag 7 is appropriate to transform \n" \
                  "the time-series into a, almost, stationary time-series"))


fig.text(1.23, -1.4,com_res, fontsize=13,
        bbox=dict(boxstyle="round", edgecolor="black", facecolor="wheat"))
  
