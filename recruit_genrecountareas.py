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

## Analyze how area and genres relate


air_area_genre_d=air_store_info_d[["air_area_name", "air_genre_name"]].dropna(subset=["air_area_name",
                                                                                                     "air_genre_name"])


# (a) Create a dataframe with all areas and genre combinations
all_areas=list(set(air_area_genre_d["air_area_name"]))
all_genres=list(set(air_area_genre_d["air_genre_name"]))

areas_array=[]
genres_array=[]

for i in range(len(all_areas)):
  temp_area=all_areas[i]
  for j in range(len(all_genres)):
    temp_genre=all_genres[j]
    
    areas_array.append(temp_area)
    genres_array.append(temp_genre)
area_genre_allcombs_d=pd.DataFrame({'air_area_name':areas_array,
                                   'air_genre_name':genres_array})

# Compute the amount of genres for each area

air_genrecount_d=air_area_genre_d.groupby(["air_area_name", "air_genre_name"]).air_genre_name.size().to_frame('genre_count').reset_index()



# Merge genrecount with all combinations
area_genrecount_allcomb_d=area_genre_allcombs_d.merge(air_genrecount_d, on=["air_area_name",
                                                                           "air_genre_name"],
                                                     how="left")

area_genrecount_allcomb_d["genre_count"]=area_genrecount_allcomb_d["genre_count"].fillna(0)

print "---"
print area_genrecount_allcomb_d.shape[0]
print len(set(areas_array))
print len(set(genres_array))
print len(set(areas_array))*len(set(genres_array))
print "---"

# (b) Create a matrix consisting of each area and its count of 
# each type of restaurant

genre_count_areas=[]
area_names=[]
genre_name_ordered=np.sort(list(set(area_genrecount_allcomb_d.sort_values(by="air_genre_name", ascending=True)["air_genre_name"])))

for a, b in area_genrecount_allcomb_d.groupby("air_area_name"):
  area_names.append(a)
  temp_genrecount=b.sort_values(by="air_genre_name", ascending=True)["genre_count"].values
  genre_count_areas.append(temp_genrecount)
  
  
genre_count_areas=np.array(genre_count_areas).T

print genre_count_areas.shape
 
print genre_name_ordered  
  
# (c) Use heatmap to analyze the amount of restaurants of certain genres
# in different areas

fig, ax = plt.subplots()

sns.heatmap(genre_count_areas, ax=ax, yticklabels=genre_name_ordered,
           cbar_kws={'label':'Amount of Restaurants'})

ax.set_title("Amount of Restaurants of Each Genre in Each Area \n Each box corresponds to one genre and one area, with the color denoting the amount of restaurants",
            fontsize=13)

y_labels=range(len(genre_name_ordered))
#ax.set_yticks(y_labels)
#ax.set_yticklabels(genre_name_ordered)
fig.subplots_adjust(right=2, top=1, bottom=-0.3)

ax.set_xticklabels([])
ax.set_xlabel("Areas", fontsize=14)

fig.set_facecolor("floralwhite")

com_res="\n".join((r"$\cdot$ " "There is one area with a lot of Cafe/Sweets - might correspond to a place where spending moderate amount \n" \
                  "of money on food/drinks is really popular, like a tourist place, shopping center, or close to perhaps schools",
                  r"$\cdot$ " "Areas that have a large amount of restaurants of one type usually have a large/moderate amount of \n" \
                  "restaurants of other types - might indicate that the underlying area is a popular place to eat/drink, for example \n" \
                  "inner cities, or areas with a large nightlife",
                  r"$\cdot$ " "As expected from the analysis of the amount of restaurants of each type, Cafe/Sweets, Izakaya, \n" \
                  "and Italian/French restaurants are spread, more or less, over all different areas. The large spread of \n" \
                  "Cafe/Sweets and Izakaya over different areas indicates that informal restaurants are attractive among Japanese people"))

fig.text(0, -0.85, com_res, fontsize=13, bbox=dict(boxstyle="round",
                                                   edgecolor="black",
                                                   facecolor="wheat"))
