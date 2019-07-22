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

## Want to analyze the types of restaurants there exists, their counts

air_store_genre_d=air_store_info_d[["air_store_id",
                                   "air_genre_name"]].dropna(subset=["air_genre_name"])

air_avgvisit_d=air_visit_d.groupby("air_store_id").agg({'visitors':'mean'}).rename(columns={'visitors':'avg_visitors'}).reset_index()

air_store_genre_avgvisit_d=air_store_genre_d.merge(air_avgvisit_d, on="air_store_id")

# (a) Compute the amount of restaurans for each genre type

air_genre_count_d=air_store_genre_avgvisit_d.groupby("air_genre_name").agg({'air_genre_name':'count',
                                                                  'avg_visitors':'mean'}).rename(columns={'air_genre_name':'genre_count',
                                                                                                          'avg_visitors':'avg_visitors_genre'}).reset_index()
air_genre_count_d.sort_values(by="genre_count", ascending=True, inplace=True)



amount_restaurants=np.sum(air_genre_count_d["genre_count"])

print air_genre_count_d.query("air_genre_name=='Asian'")

print "---"
print "Q: How many genre types exists?"
print air_genre_count_d.shape[0]
print "---"


# (b) Visualize the genre types and count of restaurants of that type

fig, ax = plt.subplots()

y_labels=range(air_genre_count_d.shape[0])
ax.barh(y_labels, air_genre_count_d["genre_count"], facecolor="royalblue",
       edgecolor="black")
ax.set_yticks(y_labels)
ax.set_yticklabels(air_genre_count_d["air_genre_name"],
                  fontsize=13)



for x, y, avg_visitors in zip(air_genre_count_d["genre_count"],
                                y_labels,
                             air_genre_count_d["avg_visitors_genre"]):
                               
  ax.text(x+3,y-0.1, str(np.round(x/float(amount_restaurants)*100, 1))+" %",
         fontweight="bold",
         color="black")
  ax.text(x+32, y-0.05, int(avg_visitors), fontweight="bold",
         color="green")



ax.set_facecolor("wheat")
ax.set_xlim((0, 240))
ax.set_xlabel("Amount of Restaurants")
ax.set_title("Amount of Restaurants in Different Genres")

com_foodtype="\n".join((r"$\cdot$ " "Izakaya: Informal Japanese Pub",
                       r"$\cdot$ " "Okonomiyaki: Japanese Savory Pancakes",
                       r"$\cdot$ " "Monja: Japanese Pan-fried Batter",
                       r"$\cdot$ " "Teppanyaki: Japanese Cuisine that uses an \n" \
                       "  iron graddle to cook food",
                       r"$\cdot$ " "Yakiniku: Japanese Grilled Meat Cuisine"))

fig.text(0.92, 0.55, com_foodtype, fontsize=13, fontweight="bold")

fig.set_facecolor("floralwhite")
genre_percentage_patch=mpatches.Patch(color="black",
                                     label="Percentage of Restaurants \nin Genre")
avg_visitors_patch=mpatches.Patch(color="green",
                                 label="Average Amount of Visitors \nfor Restaurants in Genre (daily)")

ax.legend(handles=[genre_percentage_patch, avg_visitors_patch])


com_perc_genre="\n".join((r"$\cdot$ " "Izakaya & Cafe/Sweets are among the most frequent \n" \
                                      "type of restaurants - most likely because they are both \n" \
                         "informal, and they are most likely more affordable 'restaurants', \n" \
                         "as compared to restaurants offering lunch and/or dinner",
                         r"$\cdot$ " "There is a large amount of Italian/French restaurants, \n" \
                         "which might imply that Italian and French food is popular in Japan - \n" \
                         "note that Italian restaurants might include Pizzerias",
                         r"$\cdot$ " "Western food and international cuisine seems to be less \n" \
                         "frequent - which might imply that western food, aside Italian and French, \n" \
                         "are not too popular in Japan"))
com_avg_visitors="\n".join((r"$\cdot$ " "The high amount of average visitors for karaoke party might indicate that large groups \n" \
                           "might visit karaoke parties together (e.g. friends, workplace friends, family)",
                           r"$\cdot$ " "The low average visitors for Bar/Cocktail, relative Izakaya, might indicate that informal \n" \
                           "pubs are more popular in Japan"))

box=dict(boxstyle="round", edgecolor="black", facecolor="wheat")
fig.text(0.92, 0.03, com_perc_genre, fontsize=12, bbox=box)
fig.text(-0.38, -0.23, com_avg_visitors, fontsize=12, bbox=box)
