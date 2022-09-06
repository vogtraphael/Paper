
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 18:16:15 2022

@author: r1vog
"""


import numpy as np
import matplotlib.pyplot as plt

Em_op_old = np.linspace(15.5,3.5,15)
Em_op_ren = np.linspace(5.8,0,15)
Em_emb_ren1 = np.linspace(440,200,15)
Em_emb_ren3 = np.linspace(440,-150,15)
Em_emb_new1 = np.linspace(700,500,30)
Em_emb_new3 = np.linspace(700,-300,30)

Share = np.linspace(0,75,15)

size_font = 15



plt.figure(166,dpi = 500)
plt.rc('xtick', labelsize=size_font) 
plt.rc('ytick', labelsize=size_font) 
plt.plot(np.arange(2021,2051,1),Em_emb_new1)
plt.plot(np.arange(2021,2051,1),Em_emb_new3)
plt.xlabel('Year', fontsize = size_font)
plt.ylabel(r'$em_\mathrm{emb,new,y}$ [kgCO$_\mathrm{2eq}$/m$^2$]',  fontsize = size_font)
plt.legend([r'Em. emb. new. 2050 = 500', r'Em. emb. new. 2050 = -300' ], fontsize=size_font,bbox_to_anchor =(0.85, -0.18))
plt.legend([r'$em_\mathrm{emb,new,tu}$ = 500', r'$em_\mathrm{emb,new,tl}$ = -300' ], fontsize=size_font)
#plt.title('Developement embodied emissions new dwelling',fontsize = size_font)



plt.figure(155,dpi = 500)
plt.rc('xtick', labelsize=size_font) 
plt.rc('ytick', labelsize=size_font) 
plt.plot(Share,Em_emb_ren1)
plt.plot(Share,Em_emb_ren3)
plt.xlabel('% renovated/demolished', fontsize = size_font)
plt.ylabel(r'$em_\mathrm{emb,ren,y}$ [kgCO$_\mathrm{2eq}$/m$^2$]',  fontsize = size_font)
plt.legend([r'$em_\mathrm{emb,ren,tu}$ = 200', r'$em_\mathrm{emb,ren,tl}$ = -150' ], fontsize=size_font)
#plt.title('Developement embodied emissions renovation',fontsize = size_font)


plt.figure(133,dpi = 500)
plt.rc('xtick', labelsize=size_font) 
plt.rc('ytick', labelsize=size_font) 
plt.plot(Share,Em_op_old) 
plt.xlabel('% renovated/demolished', fontsize = size_font)
plt.ylabel(r'$em_\mathrm{op,old,y}$ [kgCO$_\mathrm{2eq}$/m$^2$/a]',  fontsize = size_font)
plt.axhline(y=3.5,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.text(15, 3.6, 'Average em. op. new dwelling 2020', fontsize=size_font-2)
plt.axhline(y=15.5,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.text(15, 15.6, 'Average em. op. old dwelling stock 2020', fontsize=size_font-2)
plt.ylim(3,16.5)
#plt.legend([r'$em_\mathrm{op,old,y}$' ], fontsize=size_font)
#plt.title('Developement operational emissions building stock',fontsize = size_font)


plt.figure(dpi = 500)
plt.plot(np.arange(2021,2051,1),np.linspace(3.5,0,30)) 
plt.xlabel('Year', fontsize = size_font)
plt.ylabel(r'$em_\mathrm{op,new,y}$ [kgCO$_\mathrm{2eq}$/m$^2$/a]',  fontsize = size_font)
#plt.legend([r'$em_\mathrm{op,new,y}$' ], fontsize=size_font)
#plt.title('Developement operational emissions new dwelling',fontsize = size_font )


plt.figure(144,dpi = 500)
plt.rc('xtick', labelsize=size_font) 
plt.rc('ytick', labelsize=size_font) 
plt.plot(Share,Em_op_ren) 
plt.xlabel('% renovated/demolished', fontsize = size_font)
plt.ylabel(r'$em_\mathrm{op,ren,y}$ [kgCO$_\mathrm{2eq}$/m$^2$/a]',  fontsize = size_font)
plt.axhline(y=5.8,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.axhline(y=3.5,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.text(23, 3.6, 'Average em. op. new build. 2020', fontsize=size_font-2)
plt.text(23, 5.9, 'Average em. op. ren. 2020', fontsize=size_font-2)
plt.ylim(-0.25,6.5)
#plt.legend([r'$em_\mathrm{op,ren,y}$' ], fontsize=size_font)
#plt.title('Developement operational emissions renovated dwelling',fontsize = size_font)


plt.figure(dpi = 500)
plt.rc('xtick', labelsize=size_font) 
plt.rc('ytick', labelsize=size_font) 
plt.plot(Share,Em_op_old)
plt.plot(Share,Em_op_ren) 
plt.xlabel('% renovated/demolished', fontsize = size_font)
plt.ylabel(r'Emissions [kgCO$_\mathrm{2eq}$/m$^2$/a]',  fontsize = size_font)
plt.axhline(y=5.8,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.axhline(y=3.5,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.axhline(y=15.5,linewidth = 0.8, color = 'black', linestyle = 'dotted' )
plt.text(23, 3.6, 'Average em. op. new build. 2020', fontsize=size_font-2)
plt.text(23, 5.9, 'Average em. op. ren. 2020', fontsize=size_font-2)
plt.text(15, 15.6, 'Average em. op. old dwelling stock 2020', fontsize=size_font-2)
plt.ylim(-0.25,17)
plt.legend([r'$em_\mathrm{op,old,y}$',r'$em_\mathrm{op,ren,y}$' ], fontsize=size_font, bbox_to_anchor=(0.6, 0.7))
#plt.title('Developement operational emissions building stock and renovated dwelling',fontsize = size_font)

plt.figure(dpi = 500)
plt.plot(np.arange(2021,2051,1),np.linspace(80,80,30)) 
plt.xlabel('Year', fontsize = size_font)
plt.ylabel(r'$em_\mathrm{emb,dem,y}$ [kgCO$_\mathrm{2eq}$/m$^2$]',  fontsize = size_font)
#plt.legend([r'$em_\mathrm{emb,dem,y}$' ], fontsize=size_font)
#plt.title('Developement emissions demolition',fontsize = size_font )



