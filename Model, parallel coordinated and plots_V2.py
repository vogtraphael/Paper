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
Em_emb_new_l = 0         #emission embodied new building, kgCO2eq/m2/a lower bound
Em_emb_new_u = 418          #emission embodied new building, kgCO2eq/m2/a upper bound

Em_Emb_ren_20 = 440         #emission embodied renovation, kgCO2eq/m2
Em_Emb_ren_l = 0         #emission embodied renovation, kgCO2eq/m2
Em_Emb_ren_u = 264          #emission embodied renovation, kgCO2eq/m2

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
Rooms_Unused_u = Rooms_Unused_20                              #Total number of unused Rooms in Zürich in 2050 upper limit


#Range of variable parameter 
Em_emb_new_p = np.linspace(Em_emb_new_l,Em_emb_new_u,5)     
Em_emb_ren_p = np.linspace(Em_Emb_ren_l,Em_Emb_ren_u,5)
Rooms_Unused_p = np.round(np.linspace(Rooms_Unused_l,Rooms_Unused_u, 5),0)
Rooms_Unused_p2 = np.round(Rooms_Unused_p/1000,0)
Living_area_p = np.linspace(Living_area_l, Living_area_u, 5)
Living_density_p = np.round(np.linspace(Living_density_l, Living_density_u,5),3)
Ren_share_p = np.round(np.linspace(0.55,0.73,5),3)
t_p = np.linspace(20,30,5)

#Range of results for parallel coordinates plot
New_cons_p = np.linspace(0,1200000,5)
New_cons_p2 = New_cons_p/1000000
Em_op_old_tot_p = np.linspace(0,20,5)
Em_op_ren_tot_p = np.linspace(0,8,5)
Em_emb_ren_tot_p = np.linspace(0,30,5)
Em_op_new_tot_p = np.linspace(0,2.5,5)
Em_emb_new_tot_p = np.linspace(0,40,5)
Em_dem_tot_p = np.linspace(0,1.5,5)
Em_tot_tot_p = np.round(np.linspace(32,72,5),1)

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
    Dem_area_y = np.array([])
    Ren_area_y = np.array([])
    
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
        Dem_area_y = np.append(Dem_area_y,Dem_rate_y*S_OLD_20)
        Ren_area_y = np.append(Ren_area_y,Ren_rate_y*S_OLD_20)
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
            Em_Op_new_tot_y = np.append(Em_Op_new_tot_y,Em_Op_new_tot_y[-1])
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
    
    return Em_Op_old_tot,Em_Op_ren_tot, Em_Op_new_tot, Em_Emb_ren_tot,Em_Emb_new_tot,Em_Dem_tot,Em_tot,Pers_new_build_tot,Em_Op_old_tot_y,Em_Op_ren_tot_y,Em_Op_new_tot_y,Em_Emb_ren_tot_y,Em_Emb_new_tot_y,Em_Dem_tot_y,Em_tot_y,Pers_new_build_tot_y,Rooms_new_y,Const_new_tot_y,Const_new_tot, Ren_area_y, Dem_area_y

#Test functions
a = Calc_func(0.61, 100000, 25, 0, 0.7,30,-200)
b = Calc_func(0.73, 2300, 65.5, -400, 0.6,10,-200)

best1 = Calc_func(Living_density_u, Rooms_Unused_u, Living_area_u, Em_emb_new_l, Ren_share_p[-1],t_p[0],Em_Emb_ren_l)
best2 = Calc_func(Living_density_u, Rooms_Unused_l, Living_area_l, Em_emb_new_l, Ren_share_p[-1],t_p[0],Em_Emb_ren_l)

"""
#Create excel and write results of all possible combinations of the variable parameters in it 
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Living_density', 'Rooms_Unused', 'Living_area','Ren_share','t_p','Em_Emb_ren','Em_emb_new','Em_Op_old_tot', 'Em_Op_ren_tot', 'Em_Op_new_tot','Em_Emb_ren_tot','Em_Emb_new_tot','Em_Dem_tot','Em_tot', 'Pers_new_build_tot'])
    for z in range(0, len(Living_density_p)):
        for i in range(0, len(Rooms_Unused_p)):
            for l in range(0, len(Living_area_p)):
                for emb in range(0,len(Em_emb_new_p)):
                    for ren in range(0,len(Ren_share_p)):
                        for t_i in range (0,len(t_p)):
                            for emb_ren in range(0,len(Em_emb_ren_p)):
                                Calc_func_res = Calc_func(Living_density_p[z], Rooms_Unused_p[i], Living_area_p[l],Em_emb_new_p[emb],Ren_share_p[ren],t_p[t_i],Em_emb_ren_p[emb_ren])
                                writer.writerow([Living_density_p[z], Rooms_Unused_p[i], Living_area_p[l], Ren_share_p[ren],t_p[t_i],Em_emb_ren_p[emb_ren],Em_emb_new_p[emb],Calc_func_res[0],Calc_func_res[1],Calc_func_res[2],Calc_func_res[3],Calc_func_res[4],Calc_func_res[5],Calc_func_res[6],Calc_func_res[7]])
                   
"""

#Function for adjusting the range of the parallel coordinate plot
def plot_range(Values_p):
    list = []
    list.append(Values_p[0] - (Values_p[-1] - Values_p[0])/80)
    list.append(Values_p[-1] + (Values_p[-1] - Values_p[0])/80)
    return list

#Plot of parallel coordinates
df = pd.read_csv('output.csv')
fig = go.Figure(data=
    go.Parcoords(
        #fig.update_traces(name=<VALUE>, selector=dict(type='parcoords'))
        name = "NAME",
        visible = True,
        line = dict(color = df['Em_tot'],
                   colorscale = 'Rainbow', #Greys,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,Picnic,Rainbow,Portland,Jet,Hot,Blackbody,Earth,Electric,Viridis,Cividis.
                   showscale = True,
                   cmin = Em_tot_tot_p[0],
                   cmax = Em_tot_tot_p[-1]),
        #fig.update_traces(dimensions=list(...), selector=dict(type='parcoords'))
        dimensions = list([
                 #constraintrange = [100000,150000],
            dict(range = plot_range(Living_density_p),
                 tickvals = Living_density_p.tolist(),
                 #label = 'Dw. density <br /> [pers/rooms]', values = df['Living_density']),
                 label = '<b><i>Dw</i><sub>dens</sub> <br /> [pers/rooms]</b>', values = df['Living_density']),
            dict(range = plot_range(Rooms_Unused_p2),
                 tickvals = Rooms_Unused_p2.tolist(),
                 label = '<b><i>Rooms</i><sub>unused</sub> <br /> [k rooms]</b>', values = df['Rooms_Unused']/1000),
            dict(range = plot_range(Living_area_p),
                 tickvals = Living_area_p.tolist(),
                 label = '<b><i>Dw</i><sub>area</sub> <br /> [m<sup>2</sup>/pers]</b>', values = df['Living_area']),
            dict(range = plot_range(Ren_share_p),
                 tickvals = Ren_share_p.tolist(),
                 label = '<b><i>SH</i><sub>ren</sub> <br /> [%]</b>', values = df['Ren_share']),
            dict(range = plot_range(t_p),
                 tickvals = t_p.tolist(),
                 label = '<b><i>t</i><sub>rd</sub> <br /> [a]</b>', values = df['t_p']),            
            dict(range = plot_range(Em_emb_ren_p),
                 tickvals = Em_emb_ren_p.tolist(),
                 label = r'<b><i>em</i><sub>emb,ren</sub> <br /> [kgCO<sub>2eq</sub>/m<sup>2</sup>]</b>', values = df['Em_Emb_ren']),
            dict(range = plot_range(Em_emb_new_p),
                 tickvals = Em_emb_new_p.tolist(),
                 label = r'<b><i>em</i><sub>emb,new</sub>  <br /> [kgCO<sub>2eq</sub>/m<sup>2</sup>]</b>', values = df['Em_emb_new']),
            dict(range = plot_range(New_cons_p2),
                 tickvals = New_cons_p2.tolist(),
                 label = '<b><i>Pers</i><sub>new</sub>  <br />  [M Pers.]', values = df['Pers_new_build_tot']/1000000),
            dict(range = plot_range(Em_op_old_tot_p),
                 tickvals = Em_op_old_tot_p.tolist(),
                 label = '<b><i>EM</i><sub>op,old</sub>  <br />  [MtCO<sub>2eq</sub>]', values = df['Em_Op_old_tot']),
            dict(range = plot_range(Em_op_ren_tot_p),
                 tickvals = Em_op_ren_tot_p.tolist(),
                 label = '<b><i>EM</i><sub>op,ren</sub> <br /> [MtCO<sub>2eq</sub>]', values = df['Em_Op_ren_tot']),
            dict(range = plot_range(Em_emb_ren_tot_p),
                 tickvals = Em_emb_ren_tot_p.tolist(),
                 label = '<b><i>EM</i><sub>emb,ren</sub> <br /> [MtCO<sub>2eq</sub>]', values = df['Em_Emb_ren_tot']),
            dict(range = plot_range(Em_op_new_tot_p),
                 tickvals = Em_op_new_tot_p.tolist(),
                 label = '<b><i>EM</i><sub>op,new</sub> <br /> [MtCO<sub>2eq</sub>]', values = df['Em_Op_new_tot']),
            dict(range = plot_range(Em_emb_new_tot_p),
                 tickvals = Em_emb_new_tot_p.tolist(),
                 label = '<b><i>EM</i><sub>emb,new</sub> <br /> [MtCO<sub>2eq</sub>]', values = df['Em_Emb_new_tot']),
            dict(range = plot_range(Em_dem_tot_p),
                 tickvals = Em_dem_tot_p.tolist(),
                 label = '<b><i>EM</i><sub>emb,dem</sub> <br /> [MtCO<sub>2eq</sub>]', values = df['Em_Dem_tot']),
            dict(range = plot_range(Em_tot_tot_p),
                 tickvals = Em_tot_tot_p.tolist(),
                 ticktext = [f'{Em_tot_tot_p[0]}/{wd_p[0]}',f'{Em_tot_tot_p[1]}/{wd_p[1]}',f'{Em_tot_tot_p[2]}/{wd_p[2]}',f'{Em_tot_tot_p[3]}/{wd_p[3]}',f'{Em_tot_tot_p[4]}/{wd_p[4]}'],
                 label = '<b><i>EM</i><sub>tot</sub> / <i>WP</i><sub>EM,tot</sub> <br /> [MtCO<sub>2eq</sub>] / [℃]', values = df['Em_tot']),])
    )
)
fig.write_html('Parallel coordinated.html')#Add path to folder (X)                     

#Running time of programm
zeitende = time.time()
print(zeitende-zeitanfang)


"""
########################################################################
############           Sceniarios and Plots          #################
########################################################################
"""

#Parameters and label names
category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.05, 0.95, 6))
wp0 = round(warming_degree(0),2)
wp10 = round(warming_degree(10),2)
wp20 = round(warming_degree(20),2)
wp30 = round(warming_degree(30),2)
wp40 = round(warming_degree(40),2)
wp50 = round(warming_degree(50),2)
wp60 = round(warming_degree(60),2)
wp70 = round(warming_degree(70),2)
wp80 = round(warming_degree(80),2)
wp90 = round(warming_degree(90),2)
wp100 = round(warming_degree(100),2)
#category_names_tot = ['∑ Em. op. old','∑ Em. op. ren.','∑ Em. op. new','∑ Em. emb. ren.','∑ Em. emb. new','∑ Em. dem.','∑ Em. tot']
category_names_tot = ['$\mathit{EM_\mathrm{op,old}}$','$\mathit{EM_\mathrm{op,ren}}$','$\mathit{EM_\mathrm{op,new}}$','$\mathit{EM_\mathrm{emb,ren}}$','$\mathit{EM_\mathrm{emb,new}}$','$\mathit{EM_\mathrm{emb,dem}}$','$\mathit{EM_\mathrm{tot}}$']
#category_names = ['$\mathit{EM_\mathrm{op,new,y-1}}$','∑ Em. op. ren.','∑ Em. op. new','∑ Em. emb. ren.','∑ Em. emb. new','∑ Em. dem.']
category_names = ['$\mathit{EM_\mathrm{op,old}}$','$\mathit{EM_\mathrm{op,ren}}$','$\mathit{EM_\mathrm{op,new}}$','$\mathit{EM_\mathrm{emb,ren}}$','$\mathit{EM_\mathrm{emb,new}}$','$\mathit{EM_\mathrm{emb,dem}}$']
size_font = 13


#Scenarios
BaU = [23.943788387759763, 1.8383346296079337, 1.4318725391317124, 9.802772020799999, 18.98253880448899, 0.5338576503071925,56.53316403209559]
S1_best = Calc_func(Living_density_u, Rooms_Unused_l, Living_area_l, Em_emb_new_l, Ren_share_p[-1],t_p[0],Em_Emb_ren_l)
print('Warming potential S1_best =', warming_degree(S1_best[6]))
S2_worst = Calc_func(Living_density_l, Rooms_Unused_u, Living_area_u, Em_emb_new_u, Ren_share_p[0],t_p[-1],Em_Emb_ren_u)
print('Warming potential S2_worst =', warming_degree(S2_worst[6])) 

S1_suf = Calc_func(Living_density_u, Rooms_Unused_l, Living_area_l, 418, 0.73,30,264)
print('Warming potential S3_real1 =', warming_degree(S1_suf[6])) 
S1_insuf = Calc_func(Living_density_l, Rooms_Unused_u, Living_area_u, 418, 0.55,30,264)
print('Warming potential S4_real2 =', warming_degree(S1_insuf[6])) 
S2_suf = Calc_func(Living_density_u, Rooms_Unused_l, Living_area_l, 0, 0.73,30,0)
print('Warming potential S4_real2 =', warming_degree(S2_suf[6])) 
S2_insuf = Calc_func(Living_density_l, Rooms_Unused_u, Living_area_u, 0, 0.55,30,0)
print('Warming potential S4_real2 =', warming_degree(S2_insuf[6])) 


#Matrix for saving scenarios
ST_plots = [S1_best,S2_worst,S1_suf,S1_insuf,S2_suf,S2_insuf]

"""
#Plot of scenarios 
for i in range (0,len(ST_plots)):
    res_val = list(ST_plots[i][0:6])
    
    
    #Plot bar plot
    fig, ax = plt.subplots(dpi = 500) 
    plt.rc('xtick', labelsize=size_font) 
    plt.rc('ytick', labelsize=size_font)
    ax = fig.add_axes([0,0,1,1])
    ax.bar(category_names,res_val,color = category_colors, edgecolor = 'black')
    plt.ylim([0, 30.5])
    yticks = ax.get_yticks()[0:-1]
    ylabel1 = []
    ylabel2 = []
    for z in range(0,len(yticks)):
        ylabel1.append(f'{int(yticks[z])}')
        ylabel2.append(f'{round(warming_degree(yticks[z]),2)}')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabel1, fontsize = size_font)
    plt.ylabel("Cumulative emissions [MtCO$_\mathrm{2eq}$]", fontsize = size_font)
    ax2 = ax.twinx()
    ax2.bar(category_names,res_val,color = category_colors, edgecolor = 'black')
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabel2, fontsize = size_font)
    ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
    plt.ylabel("Warming Potential [℃]", fontsize = size_font)
    ax.tick_params(axis='x', labelrotation = 45)
    percentage = list(np.round(res_val/sum(res_val)*100,1))
    xaxis = ax.get_xticks()
    for xz in range(0,len(xaxis)):
        ax2.text(xaxis[xz],res_val[xz] + 0.1,'%s %s' % (percentage[xz],'%'), color='black', ha = 'center')
    plt.show()
    
    
    
    #plot emissions total
    fig, ax = plt.subplots(dpi = 500)
    plt.ylim([0, 2.4])
    ax.plot(np.arange(2021,2051,1),ST_plots[i][14],label = '$\mathit{EM_\mathrm{tot,y}}$') 
    ax2 = ax.twinx()
    plt.ylim([0, 2.4])
    ax2.plot(np.arange(2021,2051,1),warming_degree(np.cumsum(ST_plots[i][14])), color = 'red',linestyle = 'dotted', label = '$\mathit{WP_\mathrm{EM,tot}}$')
    plt.xlabel('Years',fontsize = size_font)
    ax.set_ylabel('Yearly emissions [MtCO$_\mathrm{2eq}$]',fontsize = size_font)
    ax2.set_ylabel('Warming Potential [℃]',fontsize = size_font)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if i == 0: # or i == 4 or i == 6:
        ax2.legend(lines + lines2, labels + labels2, loc=7,fontsize = 13)
    else: #i == 1 or i == 3 or i == 5:
        ax2.legend(lines + lines2, labels + labels2, loc=8,fontsize = 13)
    ax.set_xlabel('Year',fontsize = size_font)
    if i == 0:
        i = 0
        #plt.title('Yearly total emissions best scenario', fontsize = size_font)
    if i == 1:
        i = 1
        #plt.title('Yearly total emissions worst scenario', fontsize = size_font)
    if i == 6:
        i = 6
        #plt.title('Yearly total emissions reasonable 1.75 °C scenario', fontsize = size_font)
    plt.show()

    #plot person new construction and new construction area
    fig, ax = plt.subplots(dpi = 500)
    ax.plot(np.arange(2021,2051,1),ST_plots[i][16]/1000,color = 'green',label = '$\mathit{Rooms_\mathrm{new,y}}$') 
    ax2 = ax.twinx()
    ax2.plot(np.arange(2021,2051,1),ST_plots[i][17]/1e6, color = 'grey',linestyle = 'dotted', label = '$\mathit{Area_\mathrm{new,y}}$')
    plt.xlabel('Years',fontsize = 13)
    ax.set_ylabel('New built rooms [k rooms]',fontsize = 13)
    ax2.set_ylabel('New built area [M m$^2$]',fontsize = 13)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=7,fontsize = 13)
    ax.set_xlabel('Year',fontsize = 13)
    ax.set_ylim([0, 100])
    ax2.set_ylim([0,4])
    #plt.title('Yearly total emissions business as usual', fontsize = size_font)
    plt.show()

"""

#Comparison different scenarios
#Best/Worst
S1_p = list(S1_best[0:7])
S2_p = list(S2_worst[0:7])
X_axis = np.arange(len(category_names_tot))
fig, ax = plt.subplots(dpi = 500)  
plt.bar(X_axis - 0.2, BaU, 0.2, label = 'Business as usual', color = plt.get_cmap('tab10')(0.4),edgecolor = 'black')
plt.bar(X_axis - 0.0, S1_p, 0.2, label = 'Best scenario', color = plt.get_cmap('tab10')(0.2),edgecolor = 'black')
plt.bar(X_axis + 0.2, S2_p, 0.2, label = 'Worst scenario', color = plt.get_cmap('tab10')(0.3),edgecolor = 'black')
plt.xticks(X_axis, category_names_tot, fontsize = 10)
plt.xticks(rotation=65)
plt.ylabel("Cumulative emissions [MtCO$_\mathrm{2eq}$]", fontsize = 10)
yticks = ax.get_yticks()
ylabel1 = []
ylabel2 = []
for i in range(0,len(yticks)):
    ylabel1.append(f'{int(yticks[i])}')
    ylabel2.append(f'{round(warming_degree(yticks[i]),2)}')
ax.set_yticks(yticks)
ax.set_yticklabels(ylabel1, fontsize = 10)
plt.legend(fontsize = 10)
ax2 = ax.twinx()
ax2.set_yticks(yticks)
ax2.set_yticklabels(ylabel2, fontsize = 10)
plt.ylabel("Warming Potential [℃]", fontsize = 10)
ax.set_axisbelow(True)
ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
#plt.title('Overview cumulative emissions best/worst scenario', fontsize = 10)
plt.show()

#Realistic scenarios
S1_p = list(S1_suf[0:7])
S2_p = list(S1_insuf[0:7])
S3_p = list(S2_suf[0:7])
S4_p = list(S2_insuf[0:7])
X_axis = np.arange(len(category_names_tot))
fig, ax = plt.subplots(dpi = 500)  
plt.bar(X_axis - 0.3, BaU, 0.15, label = 'Business as usual', color = plt.get_cmap('tab10')(0.4),edgecolor = 'black')
plt.bar(X_axis - 0.15, S1_p, 0.15, label = 'S1 sufficient', color = plt.get_cmap('tab10')(0.2),edgecolor = 'black')
plt.bar(X_axis, S2_p, 0.15, label = 'S1 insufficient', color = plt.get_cmap('tab10')(0.3),edgecolor = 'black')
plt.bar(X_axis + 0.15, S3_p, 0.15, label = 'S2 sufficient',color = plt.get_cmap('tab10')(0.1),edgecolor = 'black')
plt.bar(X_axis + 0.3, S4_p, 0.15, label = 'S2 insufficient',color = plt.get_cmap('tab10')(0.0),edgecolor = 'black')
plt.xticks(X_axis, category_names_tot, fontsize = 10)
plt.xticks(rotation=65)
plt.ylabel("Cumulative emissions [MtCO$_\mathrm{2eq}$]", fontsize = 10)
yticks = ax.get_yticks()
ylabel1 = []
ylabel2 = []
for i in range(0,len(yticks)):
    ylabel1.append(f'{int(yticks[i])}')
    ylabel2.append(f'{round(warming_degree(yticks[i]),2)}')
ax.set_yticks(yticks)
ax.set_yticklabels(ylabel1, fontsize = 10)
plt.legend(fontsize = 10)
ax2 = ax.twinx()
ax2.set_yticks(yticks)
ax2.set_yticklabels(ylabel2, fontsize = 10)
plt.ylabel("Warming Potential [℃]", fontsize = 10)
ax.set_axisbelow(True)
ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
#plt.title('Overview cumulative emissions fast/slow renovation', fontsize = 10)
plt.show()




#Comparison Best/Worst Area new construction, Area renovated, Area demolished
BaU_pers = [26912467/1e6,22279027/1e6, 6321753/1e6]
Best_pers = [S1_best[18]/1e6,sum(S1_best[19])/1e6,sum(S1_best[20]/1e6)]
Worst_pers = [S2_worst[18]/1e6,sum(S2_worst[19]/1e6),sum(S2_worst[20]/1e6)]
X_axis = np.arange(3)
plt.rc('xtick', labelsize=11) 
plt.rc('ytick', labelsize=11) 
fig, ax = plt.subplots(figsize=(2, 4),dpi = 500) 
plt.xticks(rotation=45)
plt.bar(X_axis - 0.2, BaU_pers, 0.2, label = 'Business as usual', color = plt.get_cmap('tab10')(0.4),edgecolor = 'black')
plt.bar(X_axis - 0.0, Best_pers, 0.2, label = 'Most sufficient measures', color = plt.get_cmap('tab10')(0.2),edgecolor = 'black')
plt.bar(X_axis + 0.2, Worst_pers, 0.2, label = 'Most insufficient measures', color = plt.get_cmap('tab10')(0.3),edgecolor = 'black')
#plt.ylabel("Persons new dwelling [M pers.]", fontsize = 11)
plt.legend(bbox_to_anchor=(1.05, 0.5),fontsize = 11)
ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
#plt.title('Total persons new construction best/worst scenario', x=1.1, y=1.1, fontsize = 12)
ax.set_xticks([0,1,2], labels=['$\mathit{Area_\mathrm{new}}$','$\mathit{Area_\mathrm{ren}}$','$\mathit{Area_\mathrm{dem}}$'],fontsize = 11)
plt.ylabel("New built area [M m$^2$]", fontsize = 11)
plt.show()

#Emissions total overview
label1 = ['$\mathit{EM_\mathrm{tot,y}}$ S1 sufficient','$\mathit{EM_\mathrm{tot,y}}$ S1 insufficient','$\mathit{EM_\mathrm{tot,y}}$ S2 sufficient','$\mathit{EM_\mathrm{tot,y}}$ S2 insufficient']
label2 = ['$\mathit{WP_\mathrm{EM,tot}}$ S1 sufficient','$\mathit{WP_\mathrm{EM,tot}}$ S1 insufficient','$\mathit{WP_\mathrm{EM,tot}}$ S2 sufficient','$\mathit{WP_\mathrm{EM,tot}}$ S2 insufficient']
colors = [plt.get_cmap('tab10')(0.2),plt.get_cmap('tab10')(0.3),plt.get_cmap('tab10')(0.1),plt.get_cmap('tab10')(0.0)]
BaU_emissions = np.array([2.10971547, 2.09121696, 2.07299025, 2.05503534, 2.03735224,
       2.01994095, 2.00280145, 1.98593376, 1.96933787, 1.95301379,
       1.93696151, 1.92118103, 1.90567236, 1.89043549, 1.87547042,
       1.86077716, 1.8463557 , 1.83220605, 1.8183282 , 1.80472215,
       1.7913879 , 1.77832546, 1.76553483, 1.75301599, 1.74076896,
       1.72879374, 1.71709031, 1.70565869, 1.69449888, 1.68361086])

fig, ax = plt.subplots(dpi = 500)
plt.ylim([0, 2.4])
ax.plot(np.arange(2021,2051,1),BaU_emissions,label = '$\mathit{EM_\mathrm{tot,y}}$ BaU', color = plt.get_cmap('tab10')(0.4)) 
for i in range(2, len(ST_plots)):
    ax.plot(np.arange(2021,2051,1),ST_plots[i][14],label = label1[i-2], color = colors[i-2]) 
ax2 = ax.twinx()
plt.ylim([0, 2.4])
ax2.plot(np.arange(2021,2051,1),warming_degree(np.cumsum(BaU_emissions)),linestyle = 'dotted', label = '$\mathit{WP_\mathrm{EM,tot}}$ BaU', color = plt.get_cmap('tab10')(0.4))
for i in range(2, len(ST_plots)):
    ax2.plot(np.arange(2021,2051,1),warming_degree(np.cumsum(ST_plots[i][14])),linestyle = 'dotted', label = label2[i-2],color = colors[i-2])
plt.xlabel('Years',fontsize = size_font)
ax.set_ylabel('Yearly emissions [MtCO$_\mathrm{2eq}$]',fontsize = size_font)
ax2.set_ylabel('Warming Potential [℃]',fontsize = size_font)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.15, 0.98),fontsize = 13)
plt.show()


########## sensitivity analysis ################

#### Pesimistic asumption ####

SA_g = Calc_func(0.6, Rooms_Unused_20/2, 47, 250, 0.93,30,100)[0:6]
#Values base case
Living_density_b = 0.58
Rooms_Unused_b = Rooms_Unused_20*1.2
Living_area_b = 50
Ren_share_b = 0.9
t_p_b = 28
Em_emb_new_b = 350
Em_Emb_ren_b = 200
#label
#Data lower values
results_l = {
    'Base case': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Dw. density': list(Calc_func(Living_density_u, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Unused rooms': list(Calc_func(Living_density_b, Rooms_Unused_l, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Dw. area new': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_l, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Share ren.': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_p[-1],t_p_b,Em_Emb_ren_b)[0:6]),
    'Renewal build. stock': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p[0],Em_Emb_ren_b)[0:6]),
    'Em. Emb. new.': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_l, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Em. emb. ren': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_l)[0:6])
}
#Data higher values
results_h = {
    'Base case': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Dw. density': list(Calc_func(Living_density_l, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Unused rooms': list(Calc_func(Living_density_b, Rooms_Unused_u, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Dw. area new': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_u, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Share ren.': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_p[0],t_p_b,Em_Emb_ren_b)[0:6]),
    'Renewal build. stock': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p[-1],Em_Emb_ren_b)[0:6]),
    'Em. Emb. new.': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_u, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Em. emb. ren': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_u)[0:6])
}
#Get labels
labels = list(results_l.keys())
#Data lower values
data_l = np.array(list(results_l.values()))
data_cum = data_l.cumsum(axis=1)
middle_index = data_l.shape[1]//2
offsets = data_l[:, range(middle_index)].sum(axis=1) + data_l[:, middle_index]/2
#Data higher values
data_h = np.array(list(results_h.values()))
data_cum_h = data_h.cumsum(axis=1)
middle_index_h = data_h.shape[1]//2
offsets_h = data_h[:, range(middle_index_h)].sum(axis=1) + data_h[:, middle_index_h]/2
# Color Mapping
category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.05, 0.95, data_l.shape[1]))
fig, ax = plt.subplots(figsize=(10, 5),dpi = 500)
# Plot Bars
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    widths = data_h[:, i]
    starts = data_cum_h[:, i] - widths
    rects = ax.barh(labels, widths, left=starts, height=0.5,
                    label=colname, color=color,edgecolor = 'black')    
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    widths = data_l[:, i]
    starts = -data_cum[:, i]
    rects = ax.barh(labels, widths, left=starts, height=0.5,
                    color=color,edgecolor = 'black')
# Add Zero Reference Line
ax.axvline(0, linestyle='--', color='black', alpha=.25)
ax.axvline(data_l.cumsum(axis = 1)[0][5], linestyle='--', color='black', alpha=.25)
ax.axvline(-data_l.cumsum(axis = 1)[0][5], linestyle='--', color='black', alpha=.25)
# X Axis
ax.set_xlim(-70, 70)
ax.set_xticks(np.arange(-70, 71, 10))
ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))
# Y Axis
ax.invert_yaxis()
# Remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
# Ledgend
ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')
ax.set_xlabel('Emissions [MtCO$_\mathrm{2eq}$]',fontsize=13)
# Set Background Color
fig.set_facecolor('#FFFFFF')
plt.show()



#### Optimistic asumption ####

SA_g = Calc_func(0.6, Rooms_Unused_20/2, 47, 250, 0.93,30,100)[0:6]
#Values base case
Living_density_b = 0.64
Rooms_Unused_b = Rooms_Unused_20*0.8
Living_area_b = 42
Ren_share_b = 0.97
t_p_b = 20
Em_emb_new_b = 0
Em_Emb_ren_b = 0
#label
#Data lower values
results_l = {
    'Base case': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Dw. density': list(Calc_func(Living_density_u, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Unused rooms': list(Calc_func(Living_density_b, Rooms_Unused_l, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Dw. area new': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_l, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Share ren.': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_p[-1],t_p_b,Em_Emb_ren_b)[0:6]),
    'Renewal build. stock': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p[0],Em_Emb_ren_b)[0:6]),
    'Em. Emb. new.': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_l, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Em. emb. ren': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_l)[0:6])
}
#Data higher values
results_h = {
    'Base case': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Dw. density': list(Calc_func(Living_density_l, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Unused rooms': list(Calc_func(Living_density_b, Rooms_Unused_u, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Dw. area new': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_u, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Share ren.': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_p[0],t_p_b,Em_Emb_ren_b)[0:6]),
    'Renewal build. stock': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p[-1],Em_Emb_ren_b)[0:6]),
    'Em. Emb. new.': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_u, Ren_share_b,t_p_b,Em_Emb_ren_b)[0:6]),
    'Em. emb. ren': list(Calc_func(Living_density_b, Rooms_Unused_b, Living_area_b, Em_emb_new_b, Ren_share_b,t_p_b,Em_Emb_ren_u)[0:6])
}
#Get labels
labels = list(results_l.keys())
#Data lower values
data_l = np.array(list(results_l.values()))
data_cum = data_l.cumsum(axis=1)
middle_index = data_l.shape[1]//2
offsets = data_l[:, range(middle_index)].sum(axis=1) + data_l[:, middle_index]/2
#Data higher values
data_h = np.array(list(results_h.values()))
data_cum_h = data_h.cumsum(axis=1)
middle_index_h = data_h.shape[1]//2
offsets_h = data_h[:, range(middle_index_h)].sum(axis=1) + data_h[:, middle_index_h]/2
# Color Mapping
category_colors = plt.get_cmap('coolwarm_r')(    np.linspace(0.05, 0.95, data_l.shape[1]))
fig, ax = plt.subplots(figsize=(10, 5),dpi = 500)
# Plot Bars
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    widths = data_h[:, i]
    starts = data_cum_h[:, i] - widths
    rects = ax.barh(labels, widths, left=starts, height=0.5,label=colname, color=color,edgecolor = 'black')    
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    widths = data_l[:, i]
    starts = -data_cum[:, i]
    rects = ax.barh(labels, widths, left=starts, height=0.5,color=color,edgecolor = 'black')
# Add Zero Reference Line
ax.axvline(0, linestyle='--', color='black', alpha=.25)
ax.axvline(data_l.cumsum(axis = 1)[0][5], linestyle='--', color='black', alpha=.25)
ax.axvline(-data_l.cumsum(axis = 1)[0][5], linestyle='--', color='black', alpha=.25)
# X Axis
ax.set_xlim(-50, 50)
ax.set_xticks(np.arange(-50, 51, 10))
ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))
# Y Axis
ax.invert_yaxis()
# Remove spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
# Ledgend
ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')
ax.set_xlabel('Emissions [MtCO$_\mathrm{2eq}$]',fontsize=13)
# Set Background Color
fig.set_facecolor('#FFFFFF')
plt.show()




########################################################################
##########   Sceniarios sufficiency parameter analysis   ###############
########################################################################

#Parameters 
#Standart values
renrate_s = 0.92
rooms_s = Rooms_Unused_20
livarea_s = 45.5
year_s = 30
livdens_s = 0.61
embren_s = np.array([-150,200])
embnew_s = np.array([-300,500])



############Plot dependencies embodied emisions/dwelling density 
renrate = renrate_s
livdens = np.array([0.67,0.55])
embren = embren_s
embnew = embnew_s
rooms = rooms_s
livarea = livarea_s
year = year_s
Area_new,Em_r_h,Em_r_h_tot,Em_r_l_tot,X_label = [],[],[],[],[]
#X_label.append('Dw$_{dens}$,Em$_{emb,new}$,Em$_{emb,ren}$')
for j in range(0,len(embren)):
    for i in range(0,len(livdens)):
        #X_label.append("Dw$_{dens}$ = %s, Em$_{emb,new}$ = %s, Em$_{emb,ren}$ = %s" % (livdens[i], embnew[j],embren[j]))
        X_label.append("%s, %s, %s" % (livdens[i], embnew[j],embren[j]))
        Area_new.append(Calc_func(livdens[i], rooms, livarea, embnew[j], renrate,year,embren[j])[18]/1000000)
        Em_r_h.append(Calc_func(livdens[i], rooms, livarea, embnew[j], renrate,year,embren[j])[0:6])
        Em_r_h_tot.append(Calc_func(livdens[i], rooms, livarea, embnew[j], renrate,year,embren[j])[6])              
Em_r_h = np.array(Em_r_h)
Em_r_h_cum = Em_r_h.cumsum(axis=1)    
category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.05, 0.95, len(category_names)))
X_axis = np.arange(len(Em_r_h))
X_axis2 = np.arange(0,len(Em_r_h))
#X_label.append("Dw$_{dens}$, Em$_{emb,new}$, Em$_{emb,ren}$")
fig, ax = plt.subplots(figsize=(10, 8),dpi = 500)  
plt.ylim([0,80])  
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    height2 = Em_r_h[:, i]
    starts2 = Em_r_h_cum[:,i] - height2
    ax.bar(X_axis,height2,width=0.8, bottom=starts2, label=colname, color=color, edgecolor = 'black')    
plt.xticks(X_axis2, X_label)
plt.ylabel("Cumulative emissions/Warming Potential [MtCO$_\mathrm{2eq}$]/[℃]", fontsize = 13)
yticks = ax.get_yticks()
ylabel = []
for i in range(0,len(yticks)):
    ylabel.append(f'{int(yticks[i])}/{round(warming_degree(yticks[i]),2)}')
ax.set_yticks(yticks)
ax.set_yticklabels(ylabel, fontsize = 13)
#plt.xlabel("X-axis: $\mathit{DW_\mathrm{dens}}$, $\mathit{Em_\mathrm{emb,new}}$, $\mathit{Em_\mathrm{emb,new}}$",fontsize = 13)
ax2 = ax.twinx()
plt.ylim([0,70])
ax2.scatter(X_axis, Area_new,marker=(5, 2),color = 'black', label = '$\mathit{Area_\mathrm{new}}$')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax2.tick_params(axis='y', labelsize=13)
plt.ylabel("New built area [M m$^2$]", fontsize = 13)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2,fontsize = 13)
ax.set_axisbelow(True)
ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
ax2.set_xticks([0,1,2,3], labels=['$\mathit{Dw_\mathrm{dens}}$ = 0.67 \n $\mathit{em_\mathrm{emb,new}}$ = -150 \n $\mathit{em_\mathrm{emb,ren}}$ = -300','$\mathit{Dw_\mathrm{dens}}$ = 0.55 \n $\mathit{em_\mathrm{emb,new}}$ = -150 \n $\mathit{em_\mathrm{emb,ren}}$ = -300','$\mathit{Dw_\mathrm{dens}}$ = 0.67 \n $\mathit{em_\mathrm{emb,new}}$ = 500 \n $\mathit{em_\mathrm{emb,ren}}$ = 200','$\mathit{Dw_\mathrm{dens}}$ = 0.55 \n $\mathit{em_\mathrm{emb,new}}$ = 500 \n $\mathit{em_\mathrm{emb,ren}}$ = 200'],fontsize = 11,multialignment='center')
#plt.title('Impact of embodied emissions and dwelling density on cumulative emissons', fontsize = 13)
plt.show()



############Plot dependencies embodied emisions/renovation share
renrate = np.array([0.98,0.8])
livdens = livdens_s
embren = embren_s
embnew = embnew_s
rooms = rooms_s
livarea = livarea_s
year = year_s
Area_new,Em_r_h,Em_r_h_tot,Em_r_l_tot,X_label = [],[],[],[],[]
#X_label.append('Dw$_{dens}$,Em$_{emb,new}$,Em$_{emb,ren}$')
for j in range(0,len(embren)):
    for i in range(0,len(renrate)):
        #X_label.append("Dw$_{dens}$ = %s, Em$_{emb,new}$ = %s, Em$_{emb,ren}$ = %s" % (livdens[i], embnew[j],embren[j]))
        X_label.append("%s, %s, %s" % (renrate[i], embnew[j],embren[j]))
        Area_new.append(Calc_func(livdens, rooms, livarea, embnew[j], renrate[i],year,embren[j])[18]/1000000)
        Em_r_h.append(Calc_func(livdens, rooms, livarea, embnew[j], renrate[i],year,embren[j])[0:6])
        Em_r_h_tot.append(Calc_func(livdens, rooms, livarea, embnew[j], renrate[i],year,embren[j])[6])              
Em_r_h = np.array(Em_r_h)
Em_r_h_cum = Em_r_h.cumsum(axis=1)    
category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.05, 0.95, len(category_names)))
X_axis = np.arange(len(Em_r_h))
X_axis2 = np.arange(0,len(Em_r_h))
#X_label.append("Dw$_{dens}$, Em$_{emb,new}$, Em$_{emb,ren}$")
fig, ax = plt.subplots(figsize=(10, 8),dpi = 500)  
plt.ylim([0,80])  
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    height2 = Em_r_h[:, i]
    starts2 = Em_r_h_cum[:,i] - height2
    ax.bar(X_axis,height2,width=0.8, bottom=starts2, label=colname, color=color, edgecolor = 'black')    
plt.xticks(X_axis2, X_label)
plt.ylabel("Cumulative emissions/Warming Potential [MtCO$_\mathrm{2eq}$]/[℃]", fontsize = 13)
yticks = ax.get_yticks()
ylabel = []
for i in range(0,len(yticks)):
    ylabel.append(f'{int(yticks[i])}/{round(warming_degree(yticks[i]),2)}')
ax.set_yticks(yticks)
ax.set_yticklabels(ylabel, fontsize = 13)
#plt.xlabel("X-axis: $\mathit{DW_\mathrm{dens}}$, $\mathit{Em_\mathrm{emb,new}}$, $\mathit{Em_\mathrm{emb,new}}$",fontsize = 13)
ax2 = ax.twinx()
plt.ylim([0,70])
ax2.scatter(X_axis, Area_new,marker=(5, 2),color = 'black', label = '$\mathit{Area_\mathrm{new}}$')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax2.tick_params(axis='y', labelsize=13)
plt.ylabel("New built area [M m$^2$]", fontsize = 13)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2,fontsize = 13)
ax.set_axisbelow(True)
ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
ax2.set_xticks([0,1,2,3], labels=['$\mathit{SH_\mathrm{ren}}$ = 0.98 \n $\mathit{em_\mathrm{emb,new}}$ = -150 \n $\mathit{em_\mathrm{emb,ren}}$ = -300','$\mathit{SH_\mathrm{ren}}$ = 0.80 \n $\mathit{em_\mathrm{emb,new}}$ = -150 \n $\mathit{em_\mathrm{emb,ren}}$ = -300','$\mathit{SH_\mathrm{ren}}$ = 0.80 \n $\mathit{em_\mathrm{emb,new}}$ = 500 \n $\mathit{em_\mathrm{emb,ren}}$ = 200','$\mathit{SH_\mathrm{ren}}$ = 0.80 \n $\mathit{em_\mathrm{emb,new}}$ = 500 \n $\mathit{em_\mathrm{emb,ren}}$ = 200'],fontsize = 11,multialignment='center')
#plt.title('Impact of embodied emissions and renovation share on cumulative emissons', fontsize = 13)
plt.show()




############Plot dependencies embodied emisions/dwelling area
renrate = renrate_s
livdens = livdens_s
embren = embren_s
embnew = embnew_s
rooms = rooms_s
livarea = np.array([25.5,65.5])
year = year_s
Area_new,Em_r_h,Em_r_h_tot,Em_r_l_tot,X_label = [],[],[],[],[]
#X_label.append('Dw$_{dens}$,Em$_{emb,new}$,Em$_{emb,ren}$')
for j in range(0,len(embren)):
    for i in range(0,len(livarea)):
        #X_label.append("Dw$_{dens}$ = %s, Em$_{emb,new}$ = %s, Em$_{emb,ren}$ = %s" % (livdens[i], embnew[j],embren[j]))
        X_label.append("%s, %s, %s" % (livarea[i], embnew[j],embren[j]))
        Area_new.append(Calc_func(livdens, rooms, livarea[i], embnew[j], renrate,year,embren[j])[18]/1000000)
        Em_r_h.append(Calc_func(livdens, rooms, livarea[i], embnew[j], renrate,year,embren[j])[0:6])
        Em_r_h_tot.append(Calc_func(livdens, rooms, livarea[i], embnew[j], renrate,year,embren[j])[6])              
Em_r_h = np.array(Em_r_h)
Em_r_h_cum = Em_r_h.cumsum(axis=1)    
category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.05, 0.95, len(category_names)))
X_axis = np.arange(len(Em_r_h))
X_axis2 = np.arange(0,len(Em_r_h))
#X_label.append("Dw$_{dens}$, Em$_{emb,new}$, Em$_{emb,ren}$")
fig, ax = plt.subplots(figsize=(10, 8),dpi = 500)  
plt.ylim([0,80])  
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    height2 = Em_r_h[:, i]
    starts2 = Em_r_h_cum[:,i] - height2
    ax.bar(X_axis,height2,width=0.8, bottom=starts2, label=colname, color=color, edgecolor = 'black')    
plt.xticks(X_axis2, X_label)
plt.ylabel("Cumulative emissions/Warming Potential [MtCO$_\mathrm{2eq}$]/[℃]", fontsize = 13)
yticks = ax.get_yticks()
ylabel = []
for i in range(0,len(yticks)):
    ylabel.append(f'{int(yticks[i])}/{round(warming_degree(yticks[i]),2)}')
ax.set_yticks(yticks)
ax.set_yticklabels(ylabel, fontsize = 13)
#plt.xlabel("X-axis: $\mathit{DW_\mathrm{dens}}$, $\mathit{Em_\mathrm{emb,new}}$, $\mathit{Em_\mathrm{emb,new}}$",fontsize = 13)
ax2 = ax.twinx()
plt.ylim([0,70])
ax2.scatter(X_axis, Area_new,marker=(5, 2),color = 'black', label = '$\mathit{Area_\mathrm{new}}$')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax2.tick_params(axis='y', labelsize=13)
plt.ylabel("New built area [M m$^2$]", fontsize = 13)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2,fontsize = 13)
ax.set_axisbelow(True)
ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
ax2.set_xticks([0,1,2,3], labels=['$\mathit{Dw_\mathrm{area}}$ = 25.5 \n $\mathit{em_\mathrm{emb,new}}$ = -150 \n $\mathit{em_\mathrm{emb,ren}}$ = -300','$\mathit{Dw_\mathrm{area}}$ = 65.5 \n $\mathit{em_\mathrm{emb,new}}$ = -150 \n $\mathit{em_\mathrm{emb,ren}}$ = -300','$\mathit{Dw_\mathrm{area}}$ = 25.5 \n $\mathit{em_\mathrm{emb,new}}$ = 500 \n $\mathit{em_\mathrm{emb,ren}}$ = 200','$\mathit{Dw_\mathrm{area}}$ = 65.5 \n $\mathit{em_\mathrm{emb,new}}$ = 500 \n $\mathit{em_\mathrm{emb,ren}}$ = 200'],fontsize = 11,multialignment='center')
#plt.title('Impact of embodied emissions and dwelling area on cumulative emissons', fontsize = 13)
plt.show()



############Plot dependencies embodied emisions/unused rooms
renrate = renrate_s
livdens = livdens_s
embren = embren_s
embnew = embnew_s
rooms = np.array([0,rooms_s*2])
livarea = livarea_s
year = year_s
Area_new,Em_r_h,Em_r_h_tot,Em_r_l_tot,X_label = [],[],[],[],[]
#X_label.append('Dw$_{dens}$,Em$_{emb,new}$,Em$_{emb,ren}$')
for j in range(0,len(embren)):
    for i in range(0,len(rooms)):
        #X_label.append("Dw$_{dens}$ = %s, Em$_{emb,new}$ = %s, Em$_{emb,ren}$ = %s" % (livdens[i], embnew[j],embren[j]))
        X_label.append("%s, %s, %s" % (rooms[i], embnew[j],embren[j]))
        Area_new.append(Calc_func(livdens, rooms[i], livarea, embnew[j], renrate,year,embren[j])[18]/1000000)
        Em_r_h.append(Calc_func(livdens, rooms[i], livarea, embnew[j], renrate,year,embren[j])[0:6])
        Em_r_h_tot.append(Calc_func(livdens, rooms[i], livarea, embnew[j], renrate,year,embren[j])[6])              
Em_r_h = np.array(Em_r_h)
Em_r_h_cum = Em_r_h.cumsum(axis=1)    
category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.05, 0.95, len(category_names)))
X_axis = np.arange(len(Em_r_h))
X_axis2 = np.arange(0,len(Em_r_h))
#X_label.append("Dw$_{dens}$, Em$_{emb,new}$, Em$_{emb,ren}$")
fig, ax = plt.subplots(figsize=(10, 8),dpi = 500)  
plt.ylim([0,80])  
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    height2 = Em_r_h[:, i]
    starts2 = Em_r_h_cum[:,i] - height2
    ax.bar(X_axis,height2,width=0.8, bottom=starts2, label=colname, color=color, edgecolor = 'black')    
plt.xticks(X_axis2, X_label)
plt.ylabel("Cumulative emissions/Warming Potential [MtCO$_\mathrm{2eq}$]/[℃]", fontsize = 13)
yticks = ax.get_yticks()
ylabel = []
for i in range(0,len(yticks)):
    ylabel.append(f'{int(yticks[i])}/{round(warming_degree(yticks[i]),2)}')
ax.set_yticks(yticks)
ax.set_yticklabels(ylabel, fontsize = 13)
#plt.xlabel("X-axis: $\mathit{DW_\mathrm{dens}}$, $\mathit{Em_\mathrm{emb,new}}$, $\mathit{Em_\mathrm{emb,new}}$",fontsize = 13)
ax2 = ax.twinx()
plt.ylim([0,70])
ax2.scatter(X_axis, Area_new,marker=(5, 2),color = 'black', label = '$\mathit{Area_\mathrm{new}}$')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax2.tick_params(axis='y', labelsize=13)
plt.ylabel("New built area [M m$^2$]", fontsize = 13)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2,fontsize = 13)
ax.set_axisbelow(True)
ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
ax2.set_xticks([0,1,2,3], labels=['$\mathit{Rooms_\mathrm{unused}}$ = 0 \n $\mathit{em_\mathrm{emb,new}}$ = -150 \n $\mathit{em_\mathrm{emb,ren}}$ = -300','$\mathit{Rooms_\mathrm{unused}}$ = 526544 \n $\mathit{em_\mathrm{emb,new}}$ = -150 \n $\mathit{em_\mathrm{emb,ren}}$ = -300','$\mathit{Rooms_\mathrm{unused}}$ = 0 \n $\mathit{em_\mathrm{emb,new}}$ = 500 \n $\mathit{em_\mathrm{emb,ren}}$ = 200','$\mathit{Rooms_\mathrm{unused}}$ = 526544 \n $\mathit{em_\mathrm{emb,new}}$ = 500 \n $\mathit{em_\mathrm{emb,ren}}$ = 200'],fontsize = 11,multialignment='center')
#plt.title('Impact of embodied emissions and dwelling area on cumulative emissons', fontsize = 13)
plt.show()




############Plot dependencies embodied emisions/sufficiency measurea
renrate = np.array([0.98,0.8])
livdens = np.array([0.67,0.55])
embren = embren_s
embnew = embnew_s
rooms = np.array([0,rooms_s*2])
livarea = np.array([25.5,65.5])
year = year_s
Area_new,Em_r_h,Em_r_h_tot,Em_r_l_tot,X_label = [],[],[],[],[]
#X_label.append('Dw$_{dens}$,Em$_{emb,new}$,Em$_{emb,ren}$')
for j in range(0,len(embren)):
    for i in range(0,len(rooms)):
        #X_label.append("Dw$_{dens}$ = %s, Em$_{emb,new}$ = %s, Em$_{emb,ren}$ = %s" % (livdens[i], embnew[j],embren[j]))
        X_label.append("%s, %s, %s" % (rooms[i], embnew[j],embren[j]))
        Area_new.append(Calc_func(livdens[i], rooms[i], livarea[i], embnew[j], renrate[i],year,embren[j])[18]/1000000)
        Em_r_h.append(Calc_func(livdens[i], rooms[i], livarea[i], embnew[j], renrate[i],year,embren[j])[0:6])
        Em_r_h_tot.append(Calc_func(livdens[i], rooms[i], livarea[i], embnew[j], renrate[i],year,embren[j])[6])              
Em_r_h = np.array(Em_r_h)
Em_r_h_cum = Em_r_h.cumsum(axis=1)    
category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.05, 0.95, len(category_names)))
X_axis = np.arange(len(Em_r_h))
X_axis2 = np.arange(0,len(Em_r_h))
#X_label.append("Dw$_{dens}$, Em$_{emb,new}$, Em$_{emb,ren}$")
fig, ax = plt.subplots(figsize=(10, 8),dpi = 500)
plt.ylim([0,80])  
for i, (colname, color) in enumerate(zip(category_names, category_colors)):
    height2 = Em_r_h[:, i]
    starts2 = Em_r_h_cum[:,i] - height2
    ax.bar(X_axis,height2,width=0.8, bottom=starts2, label=colname, color=color, edgecolor = 'black')    
plt.xticks(X_axis2, X_label)
plt.ylabel("Cumulative emissions/Warming Potential [MtCO$_\mathrm{2eq}$]/[℃]", fontsize = 13)
yticks = ax.get_yticks()
ylabel = []
for i in range(0,len(yticks)):
    ylabel.append(f'{int(yticks[i])}/{round(warming_degree(yticks[i]),2)}')
ax.set_yticks(yticks)
ax.set_yticklabels(ylabel, fontsize = 13)
#plt.xlabel("X-axis: $\mathit{DW_\mathrm{dens}}$, $\mathit{Em_\mathrm{emb,new}}$, $\mathit{Em_\mathrm{emb,new}}$",fontsize = 13)
ax2 = ax.twinx()
plt.ylim([0,70])
ax2.scatter(X_axis, Area_new,marker=(5, 2),color = 'black', label = '$\mathit{Area_\mathrm{new}}$')
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax2.tick_params(axis='y', labelsize=13)
plt.ylabel("New built area [M m$^2$]", fontsize = 13)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2,fontsize = 13)
ax.set_axisbelow(True)
ax.yaxis.grid(linewidth = 0.6, color = 'black', linestyle = 'dotted' )
ax2.set_xticks([0,1,2,3], labels=['S5','S6','S7','S8'],fontsize = 11,multialignment='center')
#plt.title('Impact of embodied emissions and sufficiency measures', fontsize = 13)
plt.show()

