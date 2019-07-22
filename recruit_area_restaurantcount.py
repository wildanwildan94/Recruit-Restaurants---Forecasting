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


## Want to analyze areas and restaurants


air_area_d=air_store_info_d[["air_store_id", "air_area_name"]].dropna(subset=["air_area_name"])


print air_area_d.iloc[0]

# (c) Compute the amount of restaurants in each area

air_area_count_d=air_area_d.groupby("air_area_name").air_area_name.agg('count').to_frame('area_count').reset_index()
print "---"
print air_area_count_d.iloc[0]
print air_area_count_d.shape[0]
print "---"

# (d) Compute the top 10 and bottom 10 by restaurant count
area_count_bottom_d=air_area_count_d.sort_values(by="area_count", ascending=True).head(10).sort_values(by="area_count", ascending=True)
area_count_top_d=air_area_count_d.sort_values(by="area_count", ascending=False).head(10).sort_values(by="area_count", ascending=True)


# (e) Visualize top 10 and bottom 10 areas by restaurant count

fig, ax =plt.subplots(1,2)
y_labels=range(area_count_top_d.shape[0])
ax[0].barh(y_labels, area_count_top_d["area_count"], facecolor="crimson", edgecolor="black")
ax[0].set_yticks(y_labels)
ax[0].set_yticklabels(area_count_top_d["air_area_name"])
ax[0].set_title("Top 10 Areas by Amount of Restaurants in Area")

ax[1].barh(y_labels, area_count_bottom_d["area_count"], facecolor="crimson", edgecolor="black")
ax[1].set_yticks(y_labels)
ax[1].set_yticklabels(area_count_bottom_d["air_area_name"])
ax[1].set_title("Bottom 10 Areas by Amount of Restaurants in Area")

fig.subplots_adjust(right=1, left=-0.5, wspace=1)
for axes in ax.flatten():
  axes.set_xlabel("Amount of Restaurants")
  axes.set_facecolor("navajowhite")
  
  
com_areas="\n".join((r"$\cdot$ " " A lot of areas are connected to Tokyo - they are most likely subregions in Tokyo that extends to some other area, like 'Shibuya-ku Shibuya",
                    r"$\cdot$ " "The varying amount of restaurants per area indicates that there is indeed certain regions that attract restaurants more than others,\n" \
                    "like Tokyo and its subregions - which makes sense, as some subregions of a city will more attractive to visit, like the inner city of Tokyo, or close to places \n" \
                    "where people work",
                    r"$\cdot$ " "The bottom implies that some areas have almost no restaurants, but also imply that each area have at least two restaurants - note that low amount of restaurants \n" \
                     "will also imply that some areas won't have at least one restaurant of each genre"))

fig.text(-0.9, -0.3, com_areas,
        bbox=dict(boxstyle="round", edgecolor="black", facecolor="wheat"),
        fontsize=12)
fig.set_facecolor("floralwhite")
