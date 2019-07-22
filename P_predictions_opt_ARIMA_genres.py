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



visit_genre_d=air_visit_d.merge(air_store_info_d[["air_store_id", "air_genre_name"]], on="air_store_id",
                               how="left")


visit_genre_d=visit_genre_d.set_index("visit_date")

fig, ax = plt.subplots(4,4)

for i, axes in zip(range(len(genre_names)), ax.flatten()):

  # Define the current genre considered
  temp_genre_name=genre_names[i]
  
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
  train_upper_day=(last_day_data-datetime.timedelta(days=7)).strftime('%Y-%m-%d')

  
  train_dailyvisit_genre_d=dailyvisit_tempgenre_d.query("visit_date<='%s'"%train_upper_day).sort_values(by="visit_date",
                                                                                                   ascending=True)
  
  test_dailyvisit_genre_d=dailyvisit_tempgenre_d.query("visit_date>'%s'"%(train_upper_day)).sort_values(by="visit_date",
                                                                                                                       ascending=True)
  
  
  
  train_dailyvisit_genre_d["visitors_7day_diff"]=train_dailyvisit_genre_d["visitors"].diff(periods=7)
  train_dailyvisit_genre_d["visitors_7day_shift"]=train_dailyvisit_genre_d["visitors"].shift(periods=7)
  
  train_dailyvisit_genre_d=train_dailyvisit_genre_d.dropna(subset=["visitors_7day_diff",
                                                                  "visitors_7day_shift"])
  
  
  model=ARIMA(train_dailyvisit_genre_d["visitors_7day_diff"], order=(p_opt_MAE[i],0,q_opt_MAE[i]),
                 dates=train_dailyvisit_genre_d["visit_date"],
                 freq="D")
      
  fit=model.fit(disp=0)


  # Add predictions
  train_dailyvisit_genre_d["visitors_7day_diff_pred"]=fit.predict(train_dailyvisit_genre_d["visit_date"].min(),
                                                                   train_dailyvisit_genre_d["visit_date"].max()).values
  
  train_dailyvisit_genre_d["visitors_pred"]=train_dailyvisit_genre_d["visitors_7day_diff_pred"].values+train_dailyvisit_genre_d["visitors_7day_shift"].values
  
  
      
      
  test_dailyvisit_genre_d["visitors_7day_diff"]=test_dailyvisit_genre_d["visitors"].values-train_dailyvisit_genre_d.tail(7)["visitors"].values
  test_dailyvisit_genre_d["visitors_7day_shift"]=train_dailyvisit_genre_d.tail(7)["visitors"].values
  test_dailyvisit_genre_d["visitors_7day_diff_pred"]=fit.predict(test_dailyvisit_genre_d["visit_date"].min(),
                                                              test_dailyvisit_genre_d["visit_date"].max()).values
  
  test_dailyvisit_genre_d["visitors_pred"]=test_dailyvisit_genre_d["visitors_7day_diff_pred"].values+test_dailyvisit_genre_d["visitors_7day_shift"].values
  # Visualize results
  
  axes.plot(test_dailyvisit_genre_d["visit_date"],
           test_dailyvisit_genre_d["visitors_pred"], color="blue",
           label="Prediction",
           marker="s",
           markerfacecolor="blue")
  axes.plot(test_dailyvisit_genre_d["visit_date"],
           test_dailyvisit_genre_d["visitors"], color="orange",
           label="True",
           marker="s",
           markerfacecolor="orange")
  
  axes.tick_params(axis="x", rotation=30)
  #axes.set_xlabel("Date")
  #axes.set_ylabel("Visitors")
  
  axes.set_title("%s; MAE: %s; \n p=%s, q=%s"%(temp_genre_name, np.round(MAE_opt[i], 1), p_opt_MAE[i], q_opt_MAE[i]))
                   
  
  
pred_patch=mpatches.Patch(color="orange", label="Predicted")
true_patch=mpatches.Patch(color="blue", label="True")
fig.legend(handles=[true_patch, pred_patch], bbox_to_anchor=(1.4, 0.5))
fig.subplots_adjust(right=2.3, bottom=-1.3, hspace=0.8)
fig.set_facecolor("floralwhite")   
ax[3,0].set_xlabel("Date", fontsize=13)
ax[3,0].set_ylabel("Visitors", fontsize=13)
ax[3,2].axis("off")
ax[3,3].axis("off")
fig.suptitle("Optimal ARIMA Models, based on MAE, Predictions on a Single Week (Validation Dates); For Different Genres, 1 Day Average of Visitors", y=1.05, x=1.1)
com_res="Most predictions follow the trend of the true values, \n" \
"and is in a lot of cases not too far apart from the true values. \n" \
"The predicted values are pretty bad for a single date in Bar/Cocktail, \n" \
"because of some reason. \n" \
"The predictions are, in particular, good for Western Food \n" \
"and Italian/French."
                
  
fig.text(1.45, -1.3, com_res, fontsize=13, 
        bbox=dict(boxstyle="round", edgecolor="black",
                 facecolor="wheat"))
