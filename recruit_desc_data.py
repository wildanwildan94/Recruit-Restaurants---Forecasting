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


## Present all different relevant tables and their 

# (1) air_reserve_d

print "---"
print "Q: How does air_reserve_d look like?"
print air_reserve_d.iloc[0]
print "---"

# (2) hpg_reserve_d

print "---"
print "Q: How does hpg_reserve_d look like?"
print hpg_reserve_d.iloc[0]
print "---"


# (3) air_store_info_d

print "---"
print "Q: How does air_store_info_d look like?"
print air_store_info_d.iloc[0]
print "---"

# (4) air_visit_d
print "---"
print "Q: How does air_visit_d look like?"
print air_visit_d.iloc[0]
print "---"

# (5) date_info_d
print "---"
print "Q: How does date_info_d look like?"
print date_info_d.iloc[0]
print "---"

# (6) store_id_relation_d
print "---"
print "Q: How does store_id_relation_d look like?"
print store_id_relation_d.iloc[0]
print "---"


fig, ax = plt.subplots()
ax.axis("off")


com_title="Description of Datasets"
com_title_text="Want to present the available datasets and typical values"
com_title_air_reserve="AirREGI Reservations"
com_air_reserve="- Contains information of reservations done \n" \
"in the Air system"
com_air_reserve_attr="Store ID: Identification of the restaurant \n" \
                     "Visit Date: The date for the reservation \n" \
                     "Reservation Date: The date the reservation was done \n" \
                     "Visitors: The amount of spots reserved"
com_air_reserve_attr_vals="%s \n %s \n %s \n %s"%(str(air_reserve_d["air_store_id"].iloc[10]),
                                                 str(air_reserve_d["visit_datetime"].iloc[10]),
                                                 str(air_reserve_d["reserve_datetime"].iloc[10]),
                                                 str(air_reserve_d["reserve_visitors"].iloc[10]))

com_title_hpg_reserve="Hot Pepper Gourmet Reservations"
com_hpg_reserve="- Contains information of reservations done \n" \
"in the HPG system"
com_hpg_reserve_attr="Store ID: Identification of the restaurant \n" \
                     "Visit Date: The date for the reservation \n" \
                     "Reservation Date: The date the reservation was done \n" \
                     "Visitors: The amount of spots reserved"
com_hpg_reserve_attr_vals="%s \n %s \n %s \n %s"%(str(hpg_reserve_d["hpg_store_id"].iloc[10]),
                                                 str(hpg_reserve_d["visit_datetime"].iloc[10]),
                                                 str(hpg_reserve_d["reserve_datetime"].iloc[10]),
                                                 str(hpg_reserve_d["reserve_visitors"].iloc[10]))

com_title_airstore="AirREGI Store Information"
com_airstore="- Contains information of the stores in \n" \
"the air system"
com_airstore_attr="Store ID: Identification of the restaurant \n" \
                  "Store Genre: The type of restaurant \n" \
                  "Area: The area the restaurant is located at \n" \
                  "Latitude: The latitude of the restaurant's location \n"\
                  "Longitude: The longitude of the restaurant's location"
com_airstore_attr_vals="%s \n %s \n %s \n %s \n %s"%(str(air_store_info_d["air_store_id"].iloc[0]),
                                              str(air_store_info_d["air_genre_name"].iloc[0]),
                                              str(air_store_info_d["air_area_name"].iloc[0]),
                                              str(air_store_info_d["latitude"].iloc[0]),
                                              str(air_store_info_d["longitude"].iloc[0]))

com_title_airvisit="AirREGI Visitors Information"
com_airvisit="- Contains information of the visitors \n" \
"at restaurants for different dates"
com_airvisit_attr="Store ID: Identification of the restaurant \n" \
                  "Visit Date: A date the restaurant is open \n" \
                  "Visitors: The amount of visitors for a date"
com_airvisit_attr_vals="%s \n %s \n %s"%(str(air_visit_d["air_store_id"].iloc[0]),
                                        str(air_visit_d["visit_date"].iloc[0]),
                                        str(air_visit_d["visitors"].iloc[0]))

com_title_dateinfo="Date Information"
com_dateinfo="- Contains information of dates"
com_dateinfo_attr="Calendar Date: A date\n" \
                   "Day of the Week: Which day of the week \n"\
                   "Holiday: Whether the date is a holiday"
com_dateinfo_attr_vals="%s \n %s \n %s"%(str(date_info_d["calendar_date"].iloc[0]),
                                        str(date_info_d["day_of_week"].iloc[0]),
                                        str(date_info_d["holiday_flg"].iloc[0]))

fig.text(0,1, com_title, fontweight="bold", fontsize=20)
fig.text(0, 0.94, com_title_text, fontsize=14)

fig.text(0, 0.88, com_title_air_reserve, fontweight="bold",
        fontsize=18)
fig.text(0, 0.77, com_air_reserve, fontsize=14)
fig.text(0, 0.53, com_air_reserve_attr, fontsize=14, fontweight="bold")
fig.text(1, 0.53, com_air_reserve_attr_vals, fontsize=14)

fig.text(0, 0.4, com_title_hpg_reserve, fontweight="bold",
        fontsize=18)
fig.text(0, 0.28, com_hpg_reserve, fontsize=14)
fig.text(0, 0.05, com_hpg_reserve_attr, fontsize=14, fontweight="bold")
fig.text(1, 0.05, com_hpg_reserve_attr_vals, fontsize=14)

fig.text(0, -0.05, com_title_dateinfo, fontweight="bold",
        fontsize=18)
fig.text(0, -0.11, com_dateinfo, fontsize=14)
fig.text(0, -0.28, com_dateinfo_attr, fontweight="bold", fontsize=14)
fig.text(1, -0.28, com_dateinfo_attr_vals, fontsize=14)

fig.text(1.5, 0.88, com_title_airstore, fontweight="bold",
        fontsize=18)
fig.text(1.5, 0.77, com_airstore, fontsize=14)
fig.text(1.5, 0.48, com_airstore_attr, fontsize=14, fontweight="bold")
fig.text(2.5, 0.48, com_airstore_attr_vals, fontsize=14)

fig.text(1.5, 0.4, com_title_airvisit, fontweight="bold",
        fontsize=18)
fig.text(1.5, 0.28, com_airvisit, fontsize=14)
fig.text(1.5, 0.1, com_airvisit_attr, fontsize=14, fontweight="bold")
fig.text(2.5, 0.1, com_airvisit_attr_vals, fontsize=14)



fig.set_facecolor("floralwhite")
