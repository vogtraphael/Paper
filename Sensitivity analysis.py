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
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
from SALib.plotting.bar import plot as barplot
import matplotlib.pyplot as plt
zeitanfang = time.time()


"""
Parameters Reference to supplementary information
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
Living_density_l = 0.55 #Habitants per room in Zürich
Living_density_u = 0.67 #Habitants per room in Zürich

Living_area_20 = 45.5 #area per person, m2
Living_area_l = 25.5 #area per person, m2
Living_area_u = 65.5 #area per person, m2

#Dem_rooms_20 = 7962

Goal_y = 2050 #goal year
Start_y = 2021 #Start year
Ref_y = 2020 #Reference Year

Population_20=1551342 #Zürich population in 2018
Population_50=2000000 #Zürich population in 2050
Population_50_CH = 10440600
Pop = np.linspace(Population_20,Population_50,Goal_y+2-Start_y) #Jährliche Population bis 2050 bei linarer zunahme

Ren_rate_20 = 1/100 #renovation rate

Em_emb_new = 696            #emission embodied new building, kgCO2eq/m2
Em_emb_new_l = -300         #emission embodied new building, kgCO2eq/m2/a lower bound
Em_emb_new_u = 500          #emission embodied new building, kgCO2eq/m2/a upper bound

Em_Emb_ren_20 = 440         #emission embodied renovation, kgCO2eq/m2
Em_Emb_ren_l = -150         #emission embodied renovation, kgCO2eq/m2
Em_Emb_ren_u = 200          #emission embodied renovation, kgCO2eq/m2

Em_Dem = 80                #emissions demolishion, kgCO2eq/m2

Em_Op_ren_20 = 5.8          #Emission operational renovated building in 2020, kgCO2eq/m2/a
Em_Op_ren_goal = 0

Em_op_new_20 = 3.5      #emissions operational 2020, kgCO2eq/m2/a
Em_op_new_goal = 0        #emissions operational 2020, kgCO2eq/m2/a

Em_Op_old_20 = 15.5  #kgCO2/m2/y
Em_Op_old_all_ren = Em_op_new_20

#Rooms_Total_20 = sum(np.multiply(ap_tot_20, Rooms))             #Total number of Rooms in Zürich 
#Rooms_Total_20 = ap_tot_20_sum*ap_rooms
Rooms_Total_20 = 2805956
Rooms_dem = 7962
#Rooms_Inhabited_20 = sum(np.multiply(ap_inhab_20, Rooms))       #Total number of inhabited Rooms in Zürich 
#Rooms_Inhabited_20 = ap_inhab_20_sum*ap_rooms
Rooms_Inhabited_20 = Population_20/Living_density_20      #Total number of inhabited Rooms in Zürich 
#Rooms_Unused_20 = sum(np.multiply(ap_unused_20, Rooms))         #Total number of unused Rooms in Zürich
Rooms_Unused_20 = Rooms_Total_20-Rooms_Inhabited_20         #Total number of unused Rooms in Zürich
Rooms_Unused_l = 0                                         #Total number of unused Rooms in Zürich in 2050 lower limit
Rooms_Unused_u = Rooms_Unused_20*2                              #Total number of unused Rooms in Zürich in 2050 upper limit


#Range of variable parameter 
Em_emb_new_p = np.linspace(Em_emb_new_l,Em_emb_new_u,5)     
Em_emb_ren_p = np.linspace(Em_Emb_ren_l,Em_Emb_ren_u,5)
Rooms_Unused_p = np.round(np.linspace(Rooms_Unused_l,Rooms_Unused_u, 5),0)
Rooms_Unused_p2 = np.round(Rooms_Unused_p/1000,0)
Living_area_p = np.linspace(Living_area_l, Living_area_u, 5)
Living_density_p = np.round(np.linspace(Living_density_l, Living_density_u,5),3)
Ren_share_p = np.round(np.linspace(0.55,0.73,5),3)
t_p = np.linspace(10,30,5)

#Range of results for parallel coordinates plot
New_cons_p = np.linspace(0,1200000,5)
New_cons_p2 = New_cons_p/1000000
Em_op_old_tot_p = np.linspace(0,20,5)
Em_op_ren_tot_p = np.linspace(0,8,5)
Em_emb_ren_tot_p = np.linspace(0,30,5)
Em_op_new_tot_p = np.linspace(0,2.5,5)
Em_emb_new_tot_p = np.linspace(0,40,5)
Em_dem_tot_p = np.linspace(0,1.5,5)
Em_tot_tot_p = np.round(np.linspace(20,80,5),1)

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

CB_RBS_ZH_1_5 = CB_RBS_CH_1_5*Population_50/Population_50_CH        #Carbon budget residential building sector Schweiz
CB_RBS_ZH_2 = CB_RBS_CH_2*Population_50/Population_50_CH            #Carbon budget residential building sector Schweiz
 


"""
Functions
"""
#Function for calculating warming potential according to the total emissions
def warming_degree(Em_tot_f):
    a = (2-1.5)/(CB_RBS_ZH_2-CB_RBS_ZH_1_5)
    b = 1.5 - a*CB_RBS_ZH_1_5
    return a*Em_tot_f + b

#Range of warming potential for parallel coordinates plot
wd_p = np.round(warming_degree(Em_tot_tot_p),2)

#Function for calculating renovation rate
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



#Function for calculating demolition rate
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

#Function for calculating the emisisons according to the stock renewal
def lin_development_em_stock(Em_value_20, Em_value_goal, Sur_old_f):  #linear development emissions according stock renovated/demolished
    Em_y = (Em_value_20 - (Em_value_20-Em_value_goal)*(S_OLD_20-Sur_old_f)/S_OLD_20)
    return Em_y

def lin_development_em__op_old_stock(Em_value_20, Em_value_goal, Sur_old_f):  #linear development emissions according stock renovated/demolished
    if Sur_old_f >= 0.25*S_OLD_20:   
        Em_y = (Em_value_20 - (Em_value_20-Em_value_goal)*(S_OLD_20 - Sur_old_f)/(0.75*S_OLD_20))
        
        """
        print(Sur_old_f, 'Sur_old_f')
        print(S_OLD_20,'S_OLD_20')
        """
    else:
        Em_y = Em_value_goal
        #print(Em_y, 'Em_y')
    #print(Em_y,'Em_y')
    return Em_y

#Calculates values in the year xy for linear progression
def linearise (year_i, year_Goal, value_2020, value_Goal): #dynamic evolution of renovation rate till 2050
    return ((value_Goal-value_2020)/(year_Goal-Ref_y))*(year_i - Ref_y)+value_2020 

#Annual increase/decrease with linear progression  
def linearise_delta_y (year_Goal, value_2020, value_Goal): #dynamic evolution of renovation rate till 2050, yearly value
    return ((value_2020-value_Goal)/(year_Goal-Ref_y)) 


#Main function for calculating the model
def Calc_func(Living_density_f,Rooms_Unused_f,Living_area_f,Em_emb_new_f,Ren_share_f,t_f,Em_Emb_ren_f):
    
    #Sets the results to their initial value
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
    Em_Op_ren_y = 0
    Em_Op_new_y = 0
    Rooms_dem = 0
    Rooms_new = 0
    Rooms_avail = Rooms_Inhabited_20
    Living_density_0 = Living_density_20
    Pers_new_build_tot = 0
    
    #Creates arrays for saving the results
    Em_Op_old_tot_y = np.array([])
    Em_Op_ren_tot_y = np.array([]) 
    Em_Emb_ren_tot_y = np.array([])
    Em_Dem_tot_y = np.array([])
    Em_Op_new_tot_y = np.array([])
    Em_Emb_new_tot_y = np.array([])
    Em_tot_y = np.array([])
    Pers_new_build_tot_y = np.array([])
    Rooms_new_y = np.array([])
    Const_new_tot_y = np.array([])
    
    #Values for calculating renovation/demolition rate
    P_ren_tot = Ren_share_f
    E_ren = (P_ren_tot-t_f*B_ren)/(t_f**2)*2
    P_dem_tot =  round(0.75-P_ren_tot,4)
    Dem_rate_tot = 0
    Ren_rate_tot = 0
    
    #Iterate from year to year 
    for y in range(Start_y,Goal_y+1):
        year = y - Start_y +1   #number of year, starts with 1
        
        #Calculate renovation and demolition rate 
        Dem_rate_y = Func_dem_rate(year,P_dem_tot,t_f)
        Dem_apart_y = Dem_rate_y*ap_tot_20_sum #Calculate the demolished apartments
        Ren_rate_y = Func_ren_rate(year,E_ren,t_f)     
        Dem_rate_tot += Dem_rate_y
        Ren_rate_tot += Ren_rate_y
        #print(Ren_rate_y,'Ren_rate_y')
        #print(Dem_rate_y,'Dem_rate_tot')
        
        #Calculate the surfaces of the building stock 
        Sur_ren += S_OLD_20*Ren_rate_y
        Sur_dem += Dem_rate_y*S_OLD_20
        Sur_old_prev = Sur_old
        Sur_old -= (S_OLD_20*Ren_rate_y + Dem_rate_y*S_OLD_20)
        Sur_old = round(Sur_old,6)
        Sur_old_avg = (Sur_old+Sur_old_prev)/2
        
        #Calculate the yearly values of the parameters
        Living_density_y = Living_density_0 - linearise(y, Goal_y, 0, (Living_density_20-Living_density_f)/31*2)
        Living_density_0 = Living_density_y
        #Living_density_y = linearise(y, Goal_y, Living_density_20, Living_density_f)
        Living_area_y = linearise(y, Goal_y,Living_area_20,Living_area_f) 
        Em_op_ren_y = lin_development_em__op_old_stock(Em_Op_ren_20,Em_Op_ren_goal,Sur_old_avg)
        Em_op_new_y = linearise(y, Goal_y,Em_op_new_20,Em_op_new_goal)
        Em_Op_old_y = lin_development_em__op_old_stock(Em_Op_old_20,Em_op_new_20,Sur_old_avg) 
        Em_emb_new_y = linearise(y, Goal_y,Em_emb_new,Em_emb_new_f)
        Em_emb_ren_y = lin_development_em__op_old_stock(Em_Emb_ren_20,Em_Emb_ren_f,Sur_old_avg)
        Rooms_reoccu_y = linearise(y,Goal_y, 0, (Rooms_Unused_20-Rooms_Unused_f)/31*2)
         
        #Calculate yearly emissions of the stock
        Em_Op_old_tot_y = np.append(Em_Op_old_tot_y,Em_Op_old_y*Sur_old_avg/1e9)
        Em_Op_ren_tot_y = np.append(Em_Op_ren_tot_y,Em_Op_ren_y + Em_op_ren_y*S_OLD_20*Ren_rate_y*0.5/1e9)  #times 0.5 because renovation activities is consideret constant during year 
        Em_Op_ren_y += Em_op_ren_y*S_OLD_20*Ren_rate_y/1e9
        Em_Emb_ren_tot_y = np.append(Em_Emb_ren_tot_y,Ren_rate_y*S_OLD_20*Em_emb_ren_y/1e9)
        Em_Dem_tot_y = np.append(Em_Dem_tot_y,S_OLD_20*Dem_rate_y*Em_Dem/1e9)
        
        #Calculate the future construction avtivities
        Rooms_dem = Dem_apart_y*ap_rooms
        Rooms_avail +=  (Rooms_new-Rooms_dem + Rooms_reoccu_y)
        Hab_poss = Rooms_avail*Living_density_y
        Hab_new_ap = Pop[year] - Hab_poss
        if Hab_new_ap > 0:      #New buildings necessary
            Pers_new_build_tot_y = np.append(Pers_new_build_tot_y,Hab_new_ap)
            Rooms_new = Hab_new_ap/Living_density_y
            Rooms_new_y = np.append(Rooms_new_y,Rooms_new)
            Construction_new = Hab_new_ap*Living_area_y
            Const_new_tot_y = np.append(Const_new_tot_y,Construction_new)
            Em_Emb_new_tot_y = np.append(Em_Emb_new_tot_y,Construction_new*Em_emb_new_y/1e9)
            Em_Op_new_tot_y = np.append(Em_Op_new_tot_y,Em_Op_new_y + Construction_new*Em_op_new_y*0.5/1e9)
            Em_Op_new_y += Construction_new*Em_op_new_y/1e9
             
        else:       #No new buildings necessary
            Em_Emb_new_tot_y = np.append(Em_Emb_new_tot_y,0)
            Em_Op_new_tot_y = np.append(Em_Op_new_tot_y,0)
            Rooms_new_y = np.append(Rooms_new_y,0)
            Const_new_tot_y = np.append(Const_new_tot_y,0)
            Pers_new_build_tot_y = np.append(Pers_new_build_tot_y,0)
        
        #Yearly total emissions
        Em_tot_y = np.append(Em_tot_y,Em_Op_old_tot_y[-1] + Em_Op_ren_tot_y[-1] + Em_Emb_ren_tot_y[-1] + Em_Op_new_tot_y[-1] + Em_Emb_new_tot_y[-1] + Em_Dem_tot_y[-1])
    
    #Sum of the yearly results
    Em_Op_old_tot = sum(Em_Op_old_tot_y)
    Em_Op_ren_tot = sum(Em_Op_ren_tot_y)
    Em_Emb_ren_tot = sum(Em_Emb_ren_tot_y)
    Em_Op_new_tot = sum(Em_Op_new_tot_y)
    Em_Emb_new_tot = sum(Em_Emb_new_tot_y)
    Em_Dem_tot = sum(Em_Dem_tot_y)
    Em_tot = sum(Em_tot_y)
    Pers_new_build_tot = sum(Pers_new_build_tot_y)
    Const_new_tot = sum(Const_new_tot_y)
    
    #print(Dem_rate_tot, 'Dem_rate_tot')
    #print(Ren_rate_tot, 'Ren_rate_tot')
    
    return Em_Op_old_tot,Em_Op_ren_tot, Em_Op_new_tot, Em_Emb_ren_tot,Em_Emb_new_tot,Em_Dem_tot,Em_tot,Pers_new_build_tot,Em_Op_old_tot_y,Em_Op_ren_tot_y,Em_Op_new_tot_y,Em_Emb_ren_tot_y,Em_Emb_new_tot_y,Em_Dem_tot_y,Em_tot_y,Pers_new_build_tot_y,Rooms_new_y,Const_new_tot_y,Const_new_tot


#Test functions
a = Calc_func(0.61, 100000, 25, 0, 0.7,30,-200)
b = Calc_func(0.73, 2300, 65.5, -400, 0.6,10,-200)


#Plot settings 
sizefont = 15
plt.rc('xtick', labelsize=sizefont) 
plt.rc('ytick', labelsize=sizefont) 
font = {'size'   : sizefont}
plt.rc('font', **font)


#Sensitivity  analysis Scenario 1: 
problem = {
    'num_vars': 7,
    'names': ['$\mathit{Dw_\mathrm{dens}}$','$\mathit{Rooms_\mathrm{unused}}$','$\mathit{Dw_\mathrm{area}}$','$\mathit{em_\mathrm{emb,new}}$','$\mathit{SH_\mathrm{ren}}$','$\mathit{t_\mathrm{rd}}$','$\mathit{em_\mathrm{emb,ren}}$'],
    'bounds': [[0.55,0.67],
               [0, 262772],
               [25.5, 65.5],
               [0, 440],
               [0.55, 0.73],
               [20, 30],
               [0, 264]]
}
param_values = saltelli.sample(problem, 16384)
Y = np.zeros([param_values.shape[0]])


for i, X in enumerate(param_values):
    Y[i] = Calc_func(X[0],X[1],X[2],X[3],X[4],X[5],X[6])[6]   
Si = sobol.analyze(problem, Y)


#Plot
total, first, second = Si.to_df()
#barplot(total)
#barplot(total.sort_values(by=['ST'], ascending=False))
total2 = total.sort_values(by=['ST'], ascending=False)
total2.pop('ST_conf')
barplot(total2)
plt.xticks(rotation=45)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis total-order indices')
plt.legend(['Total order index'])
plt.savefig('ST_V1.png', dpi =500,bbox_inches='tight')

#barplot(first)
#barplot(first.sort_values(by=['S1'], ascending=False))
first2 = first.sort_values(by=['S1'], ascending=False)
first2.pop('S1_conf')
barplot(first2)
plt.xticks(rotation=45)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis first-order indices')
plt.legend(['First order index'])
plt.savefig('S1_V1.png', dpi =500,bbox_inches='tight')

#barplot(second)
#barplot(second.sort_values(by=['S2'], ascending=False))
second2 = second.sort_values(by=['S2'], ascending=False)
second2.pop('S2_conf')
barplot(second2)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis second-order indices')
plt.legend(['Second order index'])
plt.savefig('S2_V1.png', dpi =500,bbox_inches='tight')

"""

#Sensitivity  analysis 2: limited range of embodied emissions  
problem2 = {
    'num_vars': 7,
    'names': ['$\mathit{Dw_\mathrm{dens}}$','$\mathit{Rooms_\mathrm{unused}}$','$\mathit{Dw_\mathrm{area}}$','$\mathit{em_\mathrm{emb,new}}$','$\mathit{SH_\mathrm{ren}}$','$\mathit{t_\mathrm{rd}}$','$\mathit{em_\mathrm{emb,ren}}$'],
    'bounds': [[0.55,0.67],
               [0, 525544],
               [25.5, 65.5],
               [-100, 100],
               [0.8, 0.98],
               [10, 30],
               [-50, 50]]
}
param_values = saltelli.sample(problem2, 16384)
Y = np.zeros([param_values.shape[0]])
for i, X in enumerate(param_values):
    Y[i] = Calc_func(X[0],X[1],X[2],X[3],X[4],X[5],X[6])[6]   
Si2 = sobol.analyze(problem2, Y)
#Si.plot()
total, first, second = Si2.to_df()

#Plot
#barplot(total)
#barplot(total.sort_values(by=['ST'], ascending=False))
total2 = total.sort_values(by=['ST'], ascending=False)
total2.pop('ST_conf')
barplot(total2)
plt.xticks(rotation=45)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis total-order indices')
plt.legend(['Total order index'])
plt.savefig('ST_V2.png', dpi =500,bbox_inches='tight')

#barplot(first)
#barplot(first.sort_values(by=['S1'], ascending=False))
first2 = first.sort_values(by=['S1'], ascending=False)
first2.pop('S1_conf')
barplot(first2)
plt.xticks(rotation=45)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis first-order indices')
plt.legend(['First order index'])
plt.savefig('S1_V2.png', dpi =500,bbox_inches='tight')

#barplot(second)
#barplot(second.sort_values(by=['S2'], ascending=False))
second2 = second.sort_values(by=['S2'], ascending=False)
second2.pop('S2_conf')
barplot(second2)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis second-order indices')
plt.legend(['Second order index'])
plt.savefig('S2_V2.png', dpi =500,bbox_inches='tight')

Summe = total['ST'].sum()


#Sensitivity  analysis 3: full range of variable parameters, sensitivity analysis regarding new built area 
problem3 = {
    'num_vars': 7,
    'names': ['$\mathit{Dw_\mathrm{dens}}$','$\mathit{Rooms_\mathrm{unused}}$','$\mathit{Dw_\mathrm{area}}$','$\mathit{em_\mathrm{emb,new}}$','$\mathit{SH_\mathrm{ren}}$','$\mathit{t_\mathrm{rd}}$','$\mathit{em_\mathrm{emb,ren}}$'],
    'bounds': [[0.55,0.67],
               [0, 525544],
               [25.5, 65.5],
               [-300, 500],
               [0.8, 0.98],
               [10, 30],
               [-150, 200]]
}
param_values = saltelli.sample(problem3, 16384)
Y = np.zeros([param_values.shape[0]])


for i, X in enumerate(param_values):
    Y[i] = Calc_func(X[0],X[1],X[2],X[3],X[4],X[5],X[6])[18]   
Si = sobol.analyze(problem3, Y)


#Plots
total, first, second = Si.to_df()
#barplot(total)
#barplot(total.sort_values(by=['ST'], ascending=False))
total2 = total.sort_values(by=['ST'], ascending=False)
total2.pop('ST_conf')
barplot(total2)
plt.xticks(rotation=45)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis total-order indices')
plt.legend(['Total order index'])
plt.savefig('ST_V3.png', dpi =500,bbox_inches='tight')
#barplot(first)
#barplot(first.sort_values(by=['S1'], ascending=False))

first2 = first.sort_values(by=['S1'], ascending=False)
first2.pop('S1_conf')
barplot(first2)
plt.xticks(rotation=45)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis first-order indices')
plt.legend(['First order index'])
plt.savefig('S1_V3.png', dpi =500,bbox_inches='tight')
#barplot(second)
#barplot(second.sort_values(by=['S2'], ascending=False))

second2 = second.sort_values(by=['S2'], ascending=False)
second2.pop('S2_conf')
barplot(second2)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis second-order indices')
plt.legend(['Second order index'])
plt.savefig('S2_V3.png', dpi =500,bbox_inches='tight')

#Sensitivity  analysis 1: full range of variable parameters 
problem = {
    'num_vars': 7,
    'names': ['$\mathit{Dw_\mathrm{dens}}$','$\mathit{Rooms_\mathrm{unused}}$','$\mathit{Dw_\mathrm{area}}$','$\mathit{em_\mathrm{emb,new}}$','$\mathit{SH_\mathrm{ren}}$','$\mathit{t_\mathrm{rd}}$','$\mathit{em_\mathrm{emb,ren}}$'],
    'bounds': [[0.55,0.67],
               [0, 525544],
               [25.5, 65.5],
               [300, 500],
               [0.8, 0.98],
               [10, 30],
               [100, 200]]
}
param_values = saltelli.sample(problem, 16384)
Y = np.zeros([param_values.shape[0]])


for i, X in enumerate(param_values):
    Y[i] = Calc_func(X[0],X[1],X[2],X[3],X[4],X[5],X[6])[6]   
Si = sobol.analyze(problem, Y)


#Plot
total, first, second = Si.to_df()
#barplot(total)
#barplot(total.sort_values(by=['ST'], ascending=False))
total2 = total.sort_values(by=['ST'], ascending=False)
total2.pop('ST_conf')
barplot(total2)
plt.xticks(rotation=45)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis total-order indices')
plt.legend(['Total order index'])
plt.savefig('ST_V1.png', dpi =500,bbox_inches='tight')

#barplot(first)
#barplot(first.sort_values(by=['S1'], ascending=False))
first2 = first.sort_values(by=['S1'], ascending=False)
first2.pop('S1_conf')
barplot(first2)
plt.xticks(rotation=45)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis first-order indices')
plt.legend(['First order index'])
plt.savefig('S1_V1.png', dpi =500,bbox_inches='tight')

#barplot(second)
#barplot(second.sort_values(by=['S2'], ascending=False))
second2 = second.sort_values(by=['S2'], ascending=False)
second2.pop('S2_conf')
barplot(second2)
plt.ylabel("Sensitivity index")
#plt.title('Sensitivity analysis second-order indices')
plt.legend(['Second order index'])
plt.savefig('S2_V1.png', dpi =500,bbox_inches='tight')

"""

