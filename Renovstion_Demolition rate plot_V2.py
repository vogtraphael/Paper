# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:35:47 2022

@author: r1vog
"""

import pandas as pd
import numpy as np
import time
import csv
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.integrate import quad
zeitanfang = time.time()


"""
Parameters
"""
Rooms = (1,2,3,4,5,6)           #Number of rooms, for following calculations

ap_tot_20 = (51632,118577,223316,211098,101379,58025)           #Number of apartments in Zürich with 1,2,3,4,5,6 Rooms
ap_tot_20_ch = (295847,679360,1253156,1276309,696445,436057)    #Number of apartments in Switzerland with 1,2,3,4,5,6 Rooms
ap_inhab_20 = (40421,106332,207580,198945,96525,54533)          #Number of inhabited apartments in Zürich with 1,2,3,4,5,6 Rooms
ap_unused_20 = np.subtract(ap_tot_20,ap_inhab_20)               #Number of unused apartments in Zürich with 1,2,3,4,5,6 Rooms

ap_tot_20_sum = sum(ap_tot_20)          #Total number of apartments in Zürich 
ap_tot_20_sum_ch = sum(ap_tot_20_ch)    #Total number of apartments in Schwitzerland 
ap_inhab_20_sum = sum(ap_inhab_20)      #Total number of inhabited apartments in Zürich 
ap_unused_sum = sum(ap_unused_20)       #Total number of unused apartments in Zürich 

ap_size = 97.2      #average size of apartment in zürich, m2
ap_size_ch = 99     #average size of apartment in switzerland, m2
ap_rooms = 3.6      #average rooms per HABITED apartment in zürich

Living_density_20 = 0.61 #Habitants per room in Zürich
Living_density_l = 0.5 #Habitants per room in Zürich
Living_density_u = 0.73 #Habitants per room in Zürich

Living_area_20 = 45.5 #area per person, m2
Living_area_l = 25.5 #area per person, m2
Living_area_u = 65.5 #area per person, m2

#Dem_rooms_20 = 7962

Goal_y = 2050 #goal year
Start_y = 2021 #Start year
Ref_y = 2020 #Reference Year

Population_20=1551342 #Zürich population in 2018
Population_50=2000000 #Zürich population in 2050
Pop = np.linspace(Population_20,Population_50,Goal_y+2-Start_y) #Jährliche Population bis 2050 bei linarer zunahme

Ren_rate_20 = 1/100 #renovation rate

Em_emb_new = 696            #emission embodied new building, kgCO2eq/m2
Em_emb_new_l = -400         #emission embodied new building, kgCO2eq/m2/a lower bound
Em_emb_new_u = 900          #emission embodied new building, kgCO2eq/m2/a upper bound

Em_Emb_ren_20 = 440         #emission embodied renovation, kgCO2eq/m2
Em_Emb_ren_l = -200         #emission embodied renovation, kgCO2eq/m2
Em_Emb_ren_u = 200          #emission embodied renovation, kgCO2eq/m2

Em_Dem = 80                #emissions demolishion, kgCO2eq/m2

Em_Op_ren_20 = 5.8          #Emission operational renovated building in 2020, kgCO2eq/m2/a
Em_Op_ren_goal = 0

Em_op_new_20 = 3.5      #emissions operational 2020, kgCO2eq/m2/a
Em_op_new_goal = 0        #emissions operational 2020, kgCO2eq/m2/a

Em_Op_old_20 = 15.5  #kgCO2/m2/y
Em_Op_old_all_ren = Em_op_new_20

Rooms_Total_20 = sum(np.multiply(ap_tot_20, Rooms))             #Total number of Rooms in Zürich 
#Rooms_Total_20 = ap_tot_20_sum*ap_rooms
#Rooms_Total_20 = 2805956
Rooms_dem = 7962
#Rooms_Inhabited_20 = sum(np.multiply(ap_inhab_20, Rooms))       #Total number of inhabited Rooms in Zürich 
Rooms_Inhabited_20 = ap_inhab_20_sum*ap_rooms
Rooms_Inhabited_20 = Population_20/Living_density_20      #Total number of inhabited Rooms in Zürich 
#Rooms_Unused_20 = sum(np.multiply(ap_unused_20, Rooms))         #Total number of unused Rooms in Zürich
Rooms_Unused_20 = Rooms_Total_20-Rooms_Inhabited_20         #Total number of unused Rooms in Zürich
Rooms_Unused_l = 0                                         #Total number of unused Rooms in Zürich in 2050 lower limit
Rooms_Unused_u = Rooms_Unused_20*2                              #Total number of unused Rooms in Zürich in 2050 upper limit


# Erstellt array mit lower und upper bound
Em_emb_new_p = np.linspace(Em_emb_new_l,Em_emb_new_u,5)     #emission embodied, kgCO2eq/m2/a parametric range
Em_emb_ren_p = np.linspace(Em_Emb_ren_l,Em_Emb_ren_u,5)
Rooms_Unused_p = np.linspace(Rooms_Unused_l,Rooms_Unused_u, 5)
Living_area_p = np.linspace(Living_area_l, Living_area_u, 5)
Living_density_p = np.linspace(Living_density_l, Living_density_u,7)
Ren_share_p = np.linspace(0.55,0.73,5)
t_p = np.linspace(10,30,5)

"""
Calculation parameters
"""
#Building surface in Switzerland and in Zürich
S_OLD_20 = ap_size*ap_tot_20_sum                #surface of existing buildings in Zürich (residential)
S_OLD_20_ch = ap_size_ch*ap_tot_20_sum_ch       #surface of existing buildings in Switzerland (residential)

#Calculation of the Carbon budget for the construction sector in zürich 
CB_CH_1_5_tot = 476                 #Total carbon budget switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_tot = 1326                  #Total carbon budget switzerland for 2 degree warming, MtCO2eq.

CB_CH_1_5_op_buil_18 = CB_CH_1_5_tot*0.21                          #Carbon budget buildings operational switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_op_buil_18 = CB_CH_2_tot*0.21                              #Carbon budget buildings operational switzerland for 2 degree warming, MtCO2eq.

#CB_CH_1_5_op_buil = (-CB_CH_1_5_op_buil_18*2/32/32*2 + CB_CH_1_5_op_buil_18*2/32)*30/2                         #Carbon budget buildings operational switzerland for 1.5 degree warming, MtCO2eq.
#CB_CH_2_op_buil = (-CB_CH_2_op_buil_18*2/32/32*2 + CB_CH_2_op_buil_18*2/32)*30/2                             #Carbon budget buildings operational switzerland for 2 degree warming, MtCO2eq.

CB_CH_1_5_op_buil = CB_CH_1_5_op_buil_18-7.69-7.12-3.51-3.26            #Carbon budget residential buildings operational switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_op_buil = CB_CH_2_op_buil_18-7.69-7.12-3.51-3.26                #Carbon budget residential buildings operational switzerland for 2 degree warming, MtCO2eq.

CB_RBS_CH_1_5_op = CB_CH_1_5_op_buil*16.4/(16.4+7.5)               #Carbon budget residential buildings operational switzerland for 1.5 degree warming, MtCO2eq.
CB_RBS_CH_2_op = CB_CH_2_op_buil*16.4/(16.4+7.5)                   #Carbon budget residential buildings operational switzerland for 2 degree warming, MtCO2eq.

CB_CH_1_5_emb_dom = (CB_CH_1_5_tot*0.24-11.20-10.74)*0.4*0.3                  #Carbon budget buildings domestic embodied emissions switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_emb_dom = (CB_CH_2_tot*0.24-11.20-10.74)*0.4*0.3                      #Carbon budget buildings domestic embodied emissions switzerland for 2 degree warming, MtCO2eq.

CB_CH_1_5_emb_imp = (CB_CH_1_5_tot*0.24-11.20-10.74)*0.4*0.3/0.3*0.7          #Carbon budget buildings imported embodied emissions switzerland for 1.5 degree warming, MtCO2eq.
CB_CH_2_emb_imp = (CB_CH_2_tot*0.24-11.20-10.74)*0.4*0.3/0.3*0.7              #Carbon budget buildings imported embodied emissions switzerland for 2 degree warming, MtCO2eq.

CB_RBS_CH_1_5_emb = (CB_CH_1_5_emb_dom+CB_CH_1_5_emb_imp)*(549-542)/(812-802)       #Carbon budget residential buildings embodied emissions switzerland for 1.5 degree warming, MtCO2eq.
CB_RBS_CH_2_emb = (CB_CH_2_emb_dom+CB_CH_2_emb_imp)*(549-542)/(812-802)             #Carbon budget residential buildings embodied emissions switzerland for 2 degree warming, MtCO2eq.

CB_RBS_CH_1_5 = CB_RBS_CH_1_5_op+CB_RBS_CH_1_5_emb              #Carbon budget residential building sector Schweiz
CB_RBS_CH_2 = CB_RBS_CH_2_op+CB_RBS_CH_2_emb                    #Carbon budget residential building sector Schweiz

CB_RBS_ZH_1_5 = CB_RBS_CH_1_5*S_OLD_20/S_OLD_20_ch        #Carbon budget residential building sector Schweiz
CB_RBS_ZH_2 = CB_RBS_CH_2*S_OLD_20/S_OLD_20_ch            #Carbon budget residential building sector Schweiz
 

"""
Functions
"""
#Function renovation rate, exponential growing from 2020-2050
B_ren = Ren_rate_20
#t = 10
#Tau = 2
#E_ren = (P_ren_tot+B_ren*(Tau*np.exp(-t/Tau)-Tau))/(t+Tau*np.exp(-t/Tau)-Tau)
#E_ren = (P_ren_tot-t*B_ren)/(t**2)*2 # a von ax + b 
def Func_ren_rate(t,E_ren_f,t_f):
    t2 = t - 0.5
    if E_ren_f*t2 + B_ren > 0 and t <= t_f: 
        return E_ren_f*t2 + B_ren
    #if E_ren*t2 + B_ren <= 0 or t > t_f:
        #return E_ren_f*(1-np.exp(-t/Tau))+B_ren*np.exp(-t/Tau)
    else:
        return 0

#I1 = quad(Func_ren_rate, 0, t, args=(E_ren,t))



#Function demolition rate, linear growing from 2020-2050
B_dem = Rooms_dem/Rooms_Total_20
#a_dem = (P_dem_tot-t*B_dem)/(t**2)*2 # a von ax + b 
def Func_dem_rate(t,P_dem_tot_f,t_f):
    t2 = t-0.5
    a_dem = (P_dem_tot_f-(t_f)*B_dem)/((t_f)**2)*2 # a von ax + b 
    if P_dem_tot_f < B_dem*t_f/2:
        t_x = P_dem_tot_f/B_dem*2
        a_dem_x = -B_dem/t_x        
        if t - t_x <= 0:
            return a_dem_x*t2 + B_dem
        if t - t_x < 1:
            return (a_dem_x*(t-1+(t_x-(t-1))/2) + B_dem)*(t_x-(t-1))
        if t >= t_x:
            return 0
        
    if a_dem*t2 + B_dem > 0 and t<=t_f: 
        return a_dem*t2 + B_dem
    else:
        return 0
#I2 = quad(Func_dem_rate, 0, t,args=(a_dem,10))


def lin_development_em_stock(Em_value_20, Em_value_goal, Sur_old_f):  #linear development emissions according stock renovated/demolished
    Em_y = (Em_value_20 - (Em_value_20-Em_value_goal)*(S_OLD_20-Sur_old_f)/S_OLD_20)
    return Em_y

#Linearisieren, Wert in Jahr xy
def linearise (year_i, year_Goal, value_2020, value_Goal): #dynamic evolution of renovation rate till 2050
    return ((value_Goal-value_2020)/(year_Goal-Ref_y))*(year_i - Ref_y)+value_2020 #1% is the initial renovation rate in 2018

#Jählriche Zu/abnahme bei Linearem Verlauf  
def linearise_delta_y (year_Goal, value_2020, value_Goal): #dynamic evolution of renovation rate till 2050, yearly value
    return ((value_2020-value_Goal)/(year_Goal-Ref_y)) #1% is the initial renovation rate in 2018

def Calc_func(Living_density_f,Rooms_Unused_f,Living_area_f,Em_emb_new_f,Ren_share_f,t_f,Em_Emb_ren_f):
    Sur_dem = 0
    Sur_ren = 0
    Sur_old = S_OLD_20
    
    Em_Op_old_tot = 0
    Em_Op_ren_tot = 0 
    Em_Emb_ren_tot = 0
    Em_Dem_tot = 0
    Em_Op_new_tot = 0
    Em_Emb_new_tot = 0
    Em_tot = 0
    
    Rooms_dem = 0
    Rooms_new = 0
    Rooms_avail = Rooms_Inhabited_20
    
    Construction_new_tot = 0
    Pers_new_build_tot = 0
    
    P_ren_tot = Ren_share_f
    Em_Dem_tot = (1-P_ren_tot)*S_OLD_20*Em_Dem/1e9
    
    P_dem_tot =  round(0.75-P_ren_tot,4)
    E_ren = (P_ren_tot-t_f*B_ren)/(t_f**2)*2

    Dem_rate_tot = 0
    Ren_rate_tot = 0 
    
    Dem_rate_vec = np.array([])
    Ren_rate_vec = np.array([])
    
    for y in range(Start_y,Goal_y+1):
        year = y - Start_y +1   #number of year, starts with 1
        
        Dem_rate_y = Func_dem_rate(year,P_dem_tot,t_f)
        Dem_apart_y = Dem_rate_y*ap_tot_20_sum
        Ren_rate_y = Func_ren_rate(year,E_ren,t_f)       
        
        Sur_ren += S_OLD_20*Ren_rate_y
        Sur_dem += Dem_rate_y*S_OLD_20
        Sur_old_prev = Sur_old
        Sur_old -= (S_OLD_20*Ren_rate_y + Dem_rate_y*S_OLD_20)
        Sur_old = round(Sur_old,6)
        Sur_old_avg = (Sur_old+Sur_old_prev)/2
        
        Living_density_y = linearise(y, Goal_y, Living_density_20, Living_density_f)
        Living_area_y = linearise(y, Goal_y,Living_area_20,Living_area_f) 
        Em_op_ren_y = lin_development_em_stock(Em_Op_ren_20,Em_Op_ren_goal,Sur_old_avg)
        Em_op_new_y = linearise(y, Goal_y,Em_op_new_20,Em_op_new_goal)
        Em_Op_old_y = lin_development_em_stock(Em_Op_old_20,Em_op_new_20,Sur_old_avg) 
        Em_emb_new_y = linearise(y, Goal_y,Em_emb_new,Em_emb_new_f)
        Em_emb_ren_y = lin_development_em_stock(Em_Emb_ren_20,Em_Emb_ren_f,Sur_old_avg)
        Rooms_reoccu_y = linearise(y,Goal_y, 0, (Rooms_Unused_20-Rooms_Unused_f)/30*2)
        Dem_rate_tot += Dem_rate_y
        Ren_rate_tot += Ren_rate_y
        Dem_rate_vec = np.append(Dem_rate_vec,Dem_rate_y)
        Ren_rate_vec = np.append(Ren_rate_vec,Ren_rate_y)
        
                
        Em_Emb_ren_tot += Ren_rate_y*S_OLD_20*Em_emb_ren_y/1e9
        Em_Op_old_tot += Em_Op_old_y*Sur_old/1e9
        Em_Op_ren_tot += Em_op_ren_y*S_OLD_20*Ren_rate_y*(Goal_y-Start_y-year+0.5)/1e9
        
        Rooms_dem = Dem_apart_y*ap_rooms
        Rooms_avail +=  (Rooms_new-Rooms_dem + Rooms_reoccu_y)
        Hab_poss = Rooms_avail*Living_density_y
        Hab_new_ap = Pop[year] - Hab_poss
        if Hab_new_ap > 0:
            Pers_new_build_tot += Hab_new_ap
            Rooms_new = Hab_new_ap/Living_density_y
            Construction_new = Hab_new_ap*Living_area_y
            Construction_new_tot += Construction_new
            Em_Emb_new_tot +=  Construction_new*Em_emb_new_y/1e9
            Em_Op_new_tot += Construction_new*Em_op_new_y*(Goal_y+1-y)/1e9
    
    Em_tot = Em_Op_old_tot + Em_Op_ren_tot + Em_Emb_ren_tot + Em_Op_new_tot + Em_Emb_new_tot + Em_Dem_tot
    
    size_font = 14
    
    print(Ren_share_f)
    if Ren_share_f == 0.97:
        plt.figure(0,dpi = 500)
        plt.plot(np.arange(2021,2051,1),Dem_rate_vec*100) 
        plt.xlabel('Year', fontsize=size_font)
        plt.ylabel(r'$Dem_\mathrm{rate,y}$ [%]', fontsize=size_font)
        
        plt.figure(1,dpi = 500)
        plt.plot(np.arange(2021,2051,1),Ren_rate_vec*100) 
        plt.xlabel('Year', fontsize=size_font)
        plt.ylabel(r'$Ren_\mathrm{rate,y}$ [%]', fontsize=size_font)
    
    if Ren_share_f == 0.73:
        plt.figure(2,dpi = 500)
        plt.plot(np.arange(2021,2051,1),Dem_rate_vec*100) 
        plt.xlabel('Year', fontsize=size_font)
        plt.ylabel(r'$Dem_\mathrm{rate,y}$ [%]', fontsize=size_font)
        
        plt.figure(3,dpi = 500)
        plt.plot(np.arange(2021,2051,1),Ren_rate_vec*100) 
        plt.xlabel('Year', fontsize=size_font)
        plt.ylabel(r'$Ren_\mathrm{rate,y}$ [%]', fontsize=size_font)
    
    if Ren_share_f == 0.55:
        plt.figure(4,dpi = 500)
        plt.plot(np.arange(2021,2051,1),Dem_rate_vec*100) 
        plt.xlabel('Year', fontsize=size_font)
        plt.ylabel(r'$Dem_\mathrm{rate,y}$ [%]', fontsize=size_font)
        
        plt.figure(5,dpi = 500)
        plt.plot(np.arange(2021,2051,1),Ren_rate_vec*100) 
        plt.xlabel('Year', fontsize=size_font)
        plt.ylabel(r'$Ren_\mathrm{rate,y}$ [%]', fontsize=size_font)
        
    print(Dem_rate_tot, 'Dem_rate_tot')
    print(Ren_rate_tot, 'Ren_rate_tot')
    
    return Em_Op_old_tot,Em_Op_ren_tot, Em_Emb_ren_tot,Em_Op_new_tot,Em_Emb_new_tot,Em_Dem_tot,Em_tot,Pers_new_build_tot,Construction_new_tot


size_font = 13

a = Calc_func(0.61, 100000, 25, 0, 0.97,10,-200)
b = Calc_func(0.76, 0, 25, 0, 0.97,30,200)
c = Calc_func(0.61, 100000, 25, 0, 0.73,20,-200)
d = Calc_func(0.76, 0, 25, 0, 0.73 ,30,200)
e = Calc_func(0.61, 100000, 25, 0, 0.55,20,-200)
f = Calc_func(0.76, 0, 25, 0, 0.55,30,200)

plt.figure(0,dpi = 500)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.legend([r'$SH_\mathrm{ren}$ = 97 %, $t_\mathrm{rd}$ = 10 a', r'$SH_\mathrm{ren}$ = 97 %, $t_\mathrm{rd}$ = 30 a' ], fontsize=size_font)

plt.figure(1,dpi = 500)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.legend([r'$SH_\mathrm{ren}$ = 97 %, $t_\mathrm{rd}$ = 10 a', r'$SH_\mathrm{ren}$ = 97 %, $t_\mathrm{rd}$ = 30 a' ], fontsize=size_font)


plt.figure(2,dpi = 500)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.axhline(y=B_dem*100,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.text(2035, B_dem*100 + 0.001, 'Demolition rate 2020', fontsize=size_font-2)
plt.ylim([-0.15,2.5])
plt.legend([r'$SH_\mathrm{ren,tu}$ = 70 %, $t_\mathrm{rd,tl}$ = 20 a', r'$SH_\mathrm{ren,tu}$ = 70 %, $t_\mathrm{rd,tu}$ = 30 a' ], fontsize=size_font)

plt.figure(3,dpi = 500)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.axhline(y=1,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.text(2035, 1.1, 'Renovation rate 2020', fontsize=size_font-2)
plt.legend([r'$SH_\mathrm{ren,tu}$ = 70 %, $t_\mathrm{rd,tl}$ = 20 a', r'$SH_\mathrm{ren,tu}$ = 70 %, $t_\mathrm{rd,tu}$ = 30 a' ], fontsize=size_font)
plt.ylim([-0.85,8])
#plt.title('Developement renovation rate', fontsize =size_font+1) #fontweight="bold")


plt.figure(4,dpi = 500)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.axhline(y=B_dem*100,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.text(2035, B_dem*100 + 0.001, 'Demolition rate 2020', fontsize=size_font-2)
plt.ylim([-0.15,2.5])
plt.legend([r'$SH_\mathrm{ren,tl}$ = 55 %, $t_\mathrm{rd,tl}$ = 20 a', r'$SH_\mathrm{ren,tl}$ = 55 %, $t_\mathrm{rd,tu}$ = 30 a' ], fontsize=size_font)

plt.figure(5,dpi = 500)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.axhline(y=1,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.text(2035, 1.1, 'Renovation rate 2020', fontsize=size_font-2)
plt.legend([r'$SH_\mathrm{ren,tl}$ = 55 %, $t_\mathrm{rd,tl}$ = 20 a', r'$SH_\mathrm{ren,tl}$ = 55 %, $t_\mathrm{rd,tu}$ = 30 a' ], fontsize=size_font)
plt.ylim([-0.85,8])
#plt.title('Developement renovation rate', fontsize =size_font+1) #fontweight="bold")

