#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:50:09 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import pandas as pd
from scipy import odr
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import ascii


# CMD fitting function
def linear(theta, x):
    p0, p1 = theta
    return p0+p1*x


# Reading pickle files
sep2_list = glob.glob("d_sep2_*.pkl")
sep2_list = sorted(sep2_list)

for i in np.arange(len(sep2_list)):
    filt = sep2_list[i].split('.pkl')[0].split('d_sep2_')[1]
    df_name = "d_sep2_"+filt
    exec(df_name+" = pd.read_pickle('"+sep2_list[i]+"')")


# ----- Figure & grid setting ----- #
band = ['F435W','F606W','F814W']
color = ['F435W-F606W', 'F606W-F814W', 'F606W-F814W']

mag_lim, mag_tick = [17, 29], [18, 20, 22, 24, 26, 28]
merr_lim, merr_tick = [0., 0.5], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
cidx_lim, cidx_tick = [-0.9,3.5], [-1., 0., 1., 2., 3.]
flxr_lim, flxr_tick = [-2.0,10.0], [-2., 0., 2., 4., 6., 8., 10.]
fwhm_lim, fwhm_tick = [-4.0,20.0], [-4., 0., 4., 8., 12., 16., 20.]
mu0_lim, mu0_tick = [15, 26], [16, 18, 20, 22, 24, 26]
col_lim, col_tick = [-1.0, 2.5], [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

for i in np.arange(len(band)):
    exec("ds = d_sep2_"+band[i][1:4])

    fig = plt.figure(i+1, figsize=(11,7))
    gs = GridSpec(2, 3, left=0.08, bottom=0.12, right=0.97, top=0.97,
                  height_ratios=[1.,1.], width_ratios=[1.,1.,1.], hspace=0.10, wspace=0.30)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[0,2])
    ax6 = fig.add_subplot(gs[1,2])

    # Axis 1 : mag - magerr
    ax = ax1
    ax.set_xticks(mag_tick)
    ax.set_xticklabels([])
    ax.set_yticks(merr_tick)
    ax.set_yticklabels(merr_tick, fontsize=12.0)
    ax.set_ylabel(band[i]+' error', fontsize=12.0)
    ax.set_xlim(mag_lim)
    ax.set_ylim(merr_lim)
    ax.tick_params(width=1.5, length=8.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(width=1.5,length=5.0,which='minor')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    X = ds['mag_auto'].values
    Y = ds['merr_auto'].values
    ax.plot(X, Y, 'o', ms=2.0, color='darkgray', alpha=0.7)

    # mag_cnd = ((X > 30.0) | (Y > 1.0))    # for cosmic ray candidates

    # Axis 2 : mag - C index
    ax = ax2
    ax.set_xticks(mag_tick)
    ax.set_xticklabels([])
    ax.set_yticks(cidx_tick)
    ax.set_yticklabels(cidx_tick, fontsize=12.0)
    ax.set_ylabel('Concentration index', fontsize=12.0)
    ax.set_xlim(mag_lim)
    ax.set_ylim(cidx_lim)
    ax.tick_params(width=1.5, length=8.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(width=1.5,length=5.0,which='minor')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    X = ds['mag_auto'].values
    Y = ds['mag1'].values-ds['mag3'].values
    ax.plot(X, Y, 'o', ms=2.0, color='darkgray', alpha=0.7)

    ax.plot([mag_lim[0], 26.5], [0.55, 0.55], '--', color='blue', linewidth=1.5, alpha=0.75)
    ax.plot([26.5, 28.0], [0.55, -1.0], '--', color='blue', linewidth=1.5, alpha=0.75)

    # cidx_cnd = ((Y < 0.) | ((Y < 0.55) & (Y < (-1.55/1.5)*(X-28.0)-1.0)))    # for cosmic ray candidates

    # Axis 3 : mag - FLUX_RADIUS
    ax = ax3
    ax.set_xticks(mag_tick)
    ax.set_xticklabels(mag_tick, fontsize=12.0)
    ax.set_yticks(flxr_tick)
    ax.set_yticklabels(flxr_tick, fontsize=12.0)
    ax.set_xlabel(band[i], fontsize=12.0)
    ax.set_ylabel('FLUX_RADIUS [pixel]', fontsize=12.0)
    ax.set_xlim(mag_lim)
    ax.set_ylim(flxr_lim)
    ax.tick_params(width=1.5, length=8.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(width=1.5,length=5.0,which='minor')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    X = ds['mag_auto'].values
    Y = ds['flxrad'].values
    ax.plot(X, Y, 'o', ms=2.0, color='darkgray', alpha=0.7)

    ax.plot([flxr_lim[0], 26.5], [2.25, 2.25], '--', color='blue', linewidth=1.5, alpha=0.75)
    ax.plot([26.5, 28.0], [2.25, -2.0], '--', color='blue', linewidth=1.5, alpha=0.75)

    # flxr_cnd = ((Y < 0.) | ((Y < 2.25) & (Y < -(4.25/1.5)*(X-28.0)-2.0)))

    # Axis 4 : mag - FWHM
    ax = ax4
    ax.set_xticks(mag_tick)
    ax.set_xticklabels(mag_tick, fontsize=12.0)
    ax.set_yticks(fwhm_tick)
    ax.set_yticklabels(fwhm_tick, fontsize=12.0)
    ax.set_xlabel(band[i], fontsize=12.0)
    ax.set_ylabel('FWHM [pixel]', fontsize=12.0)
    ax.set_xlim(mag_lim)
    ax.set_ylim(fwhm_lim)
    ax.tick_params(width=1.5, length=8.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(width=1.5,length=5.0,which='minor')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    X = ds['mag_auto'].values
    Y = ds['fwhm'].values
    ax.plot(X, Y, 'o', ms=2.0, color='darkgray', alpha=0.7)

    ax.plot([fwhm_lim[0], 26.5], [3.0, 3.0], '--', color='blue', linewidth=1.5, alpha=0.75)
    ax.plot([26.5, 28.0], [3.0, -4.0], '--', color='blue', linewidth=1.5, alpha=0.75)

    # fwhm_cnd = ((Y < 0.0) | ((Y < 3.0) & (Y < -(7.0/1.5)*(X-28.0)-4.0)))

    # Axis 5 : mag - mu0
    ax = ax5
    ax.set_xticks(mag_tick)
    ax.set_xticklabels([])
    ax.set_yticks(mu0_tick)
    ax.set_yticklabels(mu0_tick, fontsize=12.0)
    ax.set_ylabel(r'$\mu_{0}~{\rm[mag~arcsec^{-2}]}$', fontsize=12.0)
    ax.set_xlim(mag_lim)
    ax.set_ylim(mu0_lim)
    ax.tick_params(width=1.5, length=8.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(width=1.5,length=5.0,which='minor')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    X = ds['mag_auto'].values
    Y = ds['mu0'].values
    ax.plot(X, Y, 'o', ms=2.0, color='darkgray', alpha=0.7) 

    ax.plot(mag_lim, np.array(mag_lim)-4.0, '--', color='blue', linewidth=1.5, alpha=0.75)

    # mu0_cnd = ((Y < 28.5) & (Y < X-4.0))

    # Axis 6 : color-magnitude diagram
    ax = ax6
    ax.set_xticks(mag_tick)
    ax.set_xticklabels(mag_tick, fontsize=12.0)
    ax.set_yticks(col_tick)
    ax.set_yticklabels(col_tick, fontsize=12.0)
    ax.set_xlabel(band[i], fontsize=12.0)
    ax.set_xlim(mag_lim)
    ax.set_ylim(col_lim)
    ax.tick_params(width=1.5, length=8.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(width=1.5,length=5.0,which='minor')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    X = ds['mag_auto'].values
    if (i < np.arange(len(band))[-1]):
        exec("col = ds['mag_auto']-d_sep2_"+band[i+1][1:4]+"['mag_auto']")
        ax.set_ylabel(band[i]+r'$-$'+band[i+1], fontsize=12.0)
    elif (i == np.arange(len(band))[-1]):
        exec("col = d_sep2_"+band[i-1][1:4]+"['mag_auto']-ds['mag_auto']")
        ax.set_ylabel(band[i-1]+r'$-$'+band[i], fontsize=12.0)
    Y = col.values
    ax.plot(X, Y, 'o', ms=2.0, color='darkgray', alpha=0.7)     

    # col_cnd = ((Y < -0.5) | (Y > 2.5))

    plt.savefig('fig_sep2_'+band[i]+'.pdf')
    plt.savefig('fig_sep2_'+band[i]+'.png', dpi=300)
    plt.close()


# ----- Foreground reddening (from NED) ----- #
A_F435W = 0.109
A_F606W = 0.075
A_F814W = 0.046
A_F110W = 0.027
A_F140W = 0.018


num_JFG2 = 12171
id_JFG2 = num_JFG2-1

mag_cnd = ((d_sep2_606['mag_auto'] < 30.0) & (d_sep2_606['merr_auto'] < 0.5) & \
           (d_sep2_814['mag_auto'] < 30.0) & (d_sep2_814['merr_auto'] < 0.5))
size_cnd = ((d_sep2_606['flxrad'] > 4.) & (d_sep2_606['fwhm'] > 4.) & \
            (d_sep2_814['flxrad'] > 4.) & (d_sep2_814['fwhm'] > 4.) & \
            (d_sep2_814['cl'] < 0.4))
# mu0_cnd = ((d_sep2_606['mu0'] > d_sep2_606['mag_auto']-4.0) & \
#            (d_sep2_814['mu0'] > d_sep2_814['mag_auto']-4.0))
col_cnd = ((d_sep2_606['mag_auto']-d_sep2_814['mag_auto'] > -1.0) & \
           (d_sep2_606['mag_auto']-d_sep2_814['mag_auto'] < 2.0))
plcmd = (1*mag_cnd + 1*size_cnd + 1*col_cnd == 3)
plcmd[829] = False    # 606 image edge
plcmd[6905] = False    # 814 too faint

f = open("plcmd.reg","w")
f.write('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" ')
f.write('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
f.write('fk5\n')
for i in np.arange(np.sum(plcmd)):
    f.write(f"circle({d_sep2_606['ra'].values[plcmd][i]:.7f},")
    f.write(f"{d_sep2_606['dec'].values[plcmd][i]:.7f},1.0"+'")\n')
f.close()


# ----- Matching member galaxies ----- #
dir_Lit = "/data/jlee/DATA/HLA/McPartland+16/MACS1752/Literature/Golovich+19/"
filename = dir_Lit+"apjsaaf88bt7_mrt.txt"
table = ascii.read(filename)
table = np.array(table)

c = 2.99792e+5  # km/s
z_mu, v_sig = 0.36479, 1186.  # km/s
z_sig = v_sig * (1.0+z_mu) / c

clu = ((table['Int'] == 12) & \
	   (table['zspec'] > z_mu-3.0*z_sig) & \
	   (table['zspec'] < z_mu+3.0*z_sig))
n_clu = np.sum(clu)

coo = SkyCoord(ra=d_sep2_606['ra'].values[plcmd]*u.degree, dec=d_sep2_606['dec'].values[plcmd]*u.degree)
catalog = SkyCoord(ra=table['RAdeg'][clu]*u.degree, dec=table['DEdeg'][clu]*u.degree)

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = coo.match_to_catalog_sky(catalog)
sep_constraint = d2d < max_sep
n_mch = np.sum(sep_constraint)
coo_matches = coo[sep_constraint]
catalog_matches = catalog[idx[sep_constraint]]

f = open("member_matched.reg","w")
g = open("sep_constraint.dat","w")
f.write('global color=cyan dashlist=8 3 width=3 font="helvetica 10 normal roman" ')
f.write('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
f.write('fk5\n')
for i in np.arange(n_mch):
    f.write(f"circle({table['RAdeg'][clu][idx[sep_constraint]][i]:.7f},")
    f.write(f"{table['DEdeg'][clu][idx[sep_constraint]][i]:.7f},8.0"+'")\n')
    g.write(f"{np.arange(len(sep_constraint))[sep_constraint][i]:d}\n")
f.close()
g.close() 


# ----- Color-magnitude diagrm (F606W, F814W) of extended sources ----- #
fig = plt.figure(10, figsize=(5,6))
gs = GridSpec(1, 1, left=0.17, bottom=0.15, right=0.95, top=0.95)

ax1 = fig.add_subplot(gs[0,0])

# Axis 1
ax = ax1
ax.set_xticks([-0.5,0.0,0.5,1.0,1.5])
ax.set_xticklabels([r'$-0.5$',0.0,0.5,1.0,1.5], fontsize=15.0)
ax.set_yticks([18.0,20.0,22.0,24.0,26.0,28.0])
ax.set_yticklabels([18.0,20.0,22.0,24.0,26.0,28.0], fontsize=15.0)
ax.set_xlabel(r'(F606W$-$F814W)$_{0}$', fontsize=15.0)
ax.set_ylabel(r'F814W$_{0}$', fontsize=15.0)
ax.set_xlim([-0.5,1.5])
ax.set_ylim([28.5,17.5])
ax.tick_params(width=1.5, length=8.0)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
ax.tick_params(width=1.5,length=5.0,which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)

X = d_sep2_606['mag_auto'].values - d_sep2_814['mag_auto'].values
Y = d_sep2_814['mag_auto'].values
X = X-(A_F606W-A_F814W)
Y = Y-A_F814W
ax.plot(X[plcmd], Y[plcmd], 'o', ms=3.0, color='darkgray', alpha=0.7)
ax.plot(X[id_JFG2], Y[id_JFG2], '*', ms=20.0, color='darkorange', alpha=0.9)
ax.plot(X[plcmd][sep_constraint], Y[plcmd][sep_constraint],
	    'o', ms=3.5, color='red', alpha=0.8)

# # Fitting
# xfit = [] ; yfit = []
# e_xfit = [] ; e_yfit = []
# for i in np.arange(12):
#     mcut = ((1*plcmd == 1) & (Y > 20.0+0.25*i) & (Y < 20.0+0.25*(i+1)) & \
#             (Y > ((17.0-29.0)/(1.0-0.5))*(X-1.05)+17.0) & \
#             (Y < ((17.0-29.0)/(1.35-1.0))*(X-1.20)+17.0))
#     xfit.append(np.median(X[mcut]))
#     yfit.append(np.median(Y[mcut]))
#     e_xfit.append(np.sqrt(np.sum(d_sep2_606['merr_auto'][mcut]**2.+d_sep2_814['merr_auto'][mcut]**2.)/np.sum(mcut)))
#     e_yfit.append(np.sqrt(np.sum(d_sep2_606['merr_auto'][mcut]**2.)/np.sum(mcut)))

# lin_mod = odr.Model(linear)
# mydata = odr.RealData(xfit, yfit, sx=e_xfit, sy=e_yfit)
# myodr = odr.ODR(mydata, lin_mod, beta0=[40.0,-20.0])
# myoutput = myodr.run()

# popt = myoutput.beta
# perr = myoutput.sd_beta

# print(popt, perr)

# x = -1.0 + (2.0 - -1.0)*np.arange(1000)/1000.
# ax.plot(xfit, yfit, 's', ms=6.0, color='blue', alpha=0.9)
# ax.plot(x, ((17.0-29.0)/(1.0-0.5))*(x-1.05)+17.0, 'k--', linewidth=2.0, alpha=0.6)
# ax.plot(x, ((17.0-29.0)/(1.35-1.0))*(x-1.20)+17.0, 'k--', linewidth=2.0, alpha=0.6)
# ax.plot(x, linear(popt, x), 'k-', linewidth=2.0, alpha=0.8)

plt.savefig('fig_sep2_cmd1.pdf')
plt.savefig('fig_sep2_cmd1.png', dpi=300)
plt.close()



# Printing the running time
print('--- %s seconds ---' %(time.time()-start_time))
