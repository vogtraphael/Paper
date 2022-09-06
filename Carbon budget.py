# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 14:21:54 2022

@author: r1vog
"""
import pandas as pd
import numpy as np
import time
import csv
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.integrate import quad


#Carbon budget 18
CB_CH_1_5_tot_18 = 476                 #Total carbon budget switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_tot_18 = 1326                  #Total carbon budget switzerland for 2 degree warming, MtCO2eq.

#Emissions of year 19 and 20
Em_19_20 = 46.09 + 43.40

#Carbon budget 20/28
CB_CH_1_5_tot_20 = CB_CH_1_5_tot_18 - Em_19_20
CB_CH_2_tot_20 = CB_CH_2_tot_18 - Em_19_20
CB_CH_1_5_tot_28 = CB_CH_1_5_tot_18 - 5*Em_19_20
CB_CH_2_tot_28 = CB_CH_2_tot_18 - 5*Em_19_20

category_colors = plt.get_cmap('viridis')(np.linspace(0.65, 1, 2))
colname = ['2018 1.5 ℃','2018 2 ℃','2020 1.5 ℃','2020 2 ℃','2028 1.5 ℃','2028 2 ℃']

X_axis = [0,1]
Vec_h = [0,0,Em_19_20,Em_19_20,5*Em_19_20,5*Em_19_20]
Vec_h2 = [CB_CH_1_5_tot_18,CB_CH_2_tot_18,CB_CH_1_5_tot_20,CB_CH_2_tot_20,CB_CH_1_5_tot_28,CB_CH_2_tot_28]

#Plot
fig, ax = plt.subplots(figsize=(4, 5),dpi = 500) 
ax.bar(colname,Vec_h,width=0.8, bottom=0, label='Emissions', color=category_colors[1], edgecolor = 'black')   
ax.bar(colname,Vec_h2,width=0.8, bottom=Vec_h, label='Carbon budget', color=category_colors[0], edgecolor = 'black')      
plt.ylabel("Carbon budget [MtCO$_\mathrm{2eq}$]", fontsize = 13)
plt.xticks(rotation=90)
ax.legend(bbox_to_anchor=(1.0, 0.7), fontsize = 13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
yticks = np.round(ax.get_yticks(),0)[0:-1]
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize = 13)
plt.xticks(fontsize = 13)
#plt.title("Carbon budget switzerland 1.5 °C and 2 °C", fontsize = 19)
#plt.xticks(rotation = 45, ha = 'right') # Rotates X-Axis Ticks by 45-degrees
plt.show()




