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


## Want to analyze the amount of reservations and visitors for restaurants,
## for a given day


# (a) Set visit_datetime as index

store_reserve_visitindex_d=store_reserve_d.set_index("visit_datetime")
air_visit_dateindex_d=air_visit_d.rename(columns={'visit_date':'visit_datetime'})
air_visit_dateindex_d=air_visit_dateindex_d.set_index("visit_datetime")

# (b) Compute the sum of daily reservations and visitors for each 
# restaurants, in intervals of one day

store_visitors_daily_d=store_reserve_visitindex_d.groupby(["air_store_id", 
                              pd.TimeGrouper(freq='d')]).sum().reset_index()

air_dailyvisits_d=air_visit_dateindex_d.groupby(["air_store_id",
                          pd.TimeGrouper(freq="d")]).sum().reset_index()



store_vis_res_d=air_dailyvisits_d.merge(store_visitors_daily_d, on=["air_store_id",
                                                                   "visit_datetime"],
                                       how="left")


store_vis_res_d["reserve_visitors"]=store_vis_res_d["reserve_visitors"].fillna(0)





# (d) Visualize the distribution of reservations and visitors



res_vis=store_visitors_daily_d["reserve_visitors"].values
quantile_cutoff_res= np.quantile(res_vis, 0.99)
print quantile_cutoff_res
res_below_quantile=[x for x in res_vis if x<quantile_cutoff_res]
res_above_quantile=[x for x in res_vis if x>=quantile_cutoff_res]
count_above_quantile_res=len(res_above_quantile)

daily_visitors=air_dailyvisits_d["visitors"].values
quantile_cutoff_vis= np.quantile(daily_visitors, 0.99)
visitors_below_quantile=[x for x in daily_visitors if x<=quantile_cutoff_vis]
visitors_above_quantile=[x for x in daily_visitors if x>quantile_cutoff_vis]
count_above_quantile_vis=len(visitors_above_quantile)

visitors_relative_reservations=store_vis_res_d["visitors"].values-store_vis_res_d["reserve_visitors"].values
quantile_lower_visrelres=np.quantile(visitors_relative_reservations, 0.01)
quantile_upper_visrelres=np.quantile(visitors_relative_reservations, 0.99)
visrelres_in_quantile_region=[x for x in visitors_relative_reservations if x>quantile_lower_visrelres and x<=quantile_upper_visrelres]




fig, ax = plt.subplots(1,3)




ax[0].hist(res_below_quantile, bins=15, edgecolor="black", facecolor="sandybrown")
ax[0].set_facecolor("navajowhite")
ax[0].set_xlabel("Reservations", fontsize=13)
ax[0].set_title("Amount of Reservations", fontsize=13)
ax[0].set_ylabel("Count", fontsize=13)


"""
above_quantile_cutoff_patch_res=mpatches.Patch(color="black", label="%s Amount of Days \nwith more than %s reservations"%(count_above_quantile_res,
                                                                                                                      int(quantile_cutoff_res)))
com_above_quantile_res="%s Amount of Days \nwith more than %s reservations"%(count_above_quantile_res,
                                                                        int(quantile_cutoff_res))
ax[0,0].set_title("Reservations for a restaurant, for a given day \n Observations with more than %s (99%%-quantile) are discarded"%int(quantile_cutoff))
fig.text(0.3, 0.7, com_above_quantile_res, bbox=dict(boxstyle="round",
                                                edgecolor="black",
                                                facecolor="wheat"),
        fontsize=14)
ax[0,0].set_facecolor("navajowhite")
fig.set_facecolor("floralwhite")
ax[0,0].set_xlabel("Amount of Reservations")


"""
ax[1].hist(visitors_below_quantile, bins=15, facecolor="sandybrown",
       edgecolor="black")

ax[1].set_facecolor("wheat")
ax[1].set_xlabel("Visitors", fontsize=13)
ax[1].set_title("Amount of Visitors", fontsize=13)


ax[2].hist(visrelres_in_quantile_region, edgecolor="black",
            facecolor="sandybrown", bins=15)
ax[2].set_facecolor("wheat")
ax[2].set_xlabel("Unscheduled Visitors", fontsize=13)
ax[2].set_title("Amount of Visitors with \nNo Underlying Reservations", fontsize=13)





fig.suptitle("Histogram of reservations, visitors, and unplanned visitors for restaurants, for a given day \n Outliers removed", y=1.2,
            fontsize=14, x=1.1)

fig.subplots_adjust(right=2, bottom=0.2, top=1, wspace=0.2)

   
com_res="\n".join((r"$\cdot$ " "Amount of reservations done for a restaurant, for a given day, is often smaller than 10 but there exists \n" \
                  " cases where reservations reach larger values like 10, 20, 30 - perhaps they corresponds to holiday/weekend days" ,
                  "$\cdot$ " "Amount of visitors, generally, exceeds the number of reservations, implying that restaurants often receive \n" \
                  "more customers than the amount of reservations - indicates consumers seldom have to worry about a restaurant running out of places",
                  r"$\cdot$ " "Further illustrating the relation between reservations and visitors, the amount of visitors with no underlying \n" \
                  "reservations are mostly non-negative, implying again that restaurants tends to receive more customers than the amount of reservations they receive",
                  r"$\cdot$ " "Also note that from the amount of reservations and visitors, we can see that there exists a lot of cases where \n" \
                  "a restaurant may receive visitors without a single reservation, for a particular day - which might corresponds to weekdays \n" \
                  "or other low-traffic inducing days", 
                  r"$\cdot$ " "Days with high amount of visitors most likely correspond to weekends or holidays, which might explain \n" \
                  "why these cases occur much less than days with low amount of visitors (which probably, in turn, corresponds to weekdays)"))


fig.text(0.1, -0.55, com_res, bbox=dict(boxstyle="round", edgecolor="black",
                                     facecolor="wheat"),
        fontsize=12)
fig.set_facecolor("floralwhite")
