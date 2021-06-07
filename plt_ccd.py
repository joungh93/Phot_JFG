#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:53:48 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os, copy
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import scipy.ndimage as ndimage
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

current_dir = os.getcwd()
os.chdir("fsps_models/")
import init_hstacs_mist_test as ini
os.chdir(current_dir)


# ----- Basic paramteres ----- #
redshift = 0.3527  # McPartland+16 Table 1
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
dist_lum = cosmo.luminosity_distance(redshift).value  # Mpc
dist_mod = 5.0*np.log10(dist_lum*1.0e+6 / 10.)

A_F435W = 0.109
A_F606W = 0.075
A_F814W = 0.046
A_F110W = 0.027
A_F140W = 0.018


# CMD fitting function
def linear(theta, x):
    p0, p1 = theta
    return p0+p1*x


# ----- Reading pickle files ----- #
pht_list = glob.glob("df_pht_*.pkl")
pht_list = sorted(pht_list)
for i in np.arange(len(pht_list)):
    filt = pht_list[i].split('.pkl')[0].split('df_pht_')[1]
    df_name = "df_pht_"+filt
    exec(df_name+" = pd.read_pickle('"+pht_list[i]+"')")

sep2_list = glob.glob("d_sep2_*.pkl")
sep2_list = sorted(sep2_list)
for i in np.arange(len(sep2_list)):
    filt = sep2_list[i].split('.pkl')[0].split('d_sep2_')[1]
    df_name = "d_sep2_"+filt
    exec(df_name+" = pd.read_pickle('"+sep2_list[i]+"')")


# ----- Color-magnitude diagrm (F606W, F814W) of extended sources ----- #
num_JFG2 = 12171
id_JFG2 = num_JFG2-1

mag_cnd = ((d_sep2_606['mag_auto'] < 30.0) & (d_sep2_606['merr_auto'] < 0.5) & \
           (d_sep2_814['mag_auto'] < 30.0) & (d_sep2_814['merr_auto'] < 0.5))
size_cnd = ((d_sep2_606['flxrad'] > 4.) & (d_sep2_606['fwhm'] > 4.) & \
            (d_sep2_814['flxrad'] > 4.) & (d_sep2_814['fwhm'] > 4.) & \
            (d_sep2_814['cl'] < 0.4))
col_cnd = ((d_sep2_606['mag_auto']-d_sep2_814['mag_auto'] > -1.0) & \
           (d_sep2_606['mag_auto']-d_sep2_814['mag_auto'] < 2.0))
plcmd = (1*mag_cnd + 1*size_cnd + 1*col_cnd == 3)
plcmd[829] = False    # 606 image edge
plcmd[6905] = False    # 814 too faint

knot_cnd = ((np.isnan(df_pht_110['mag'].values) == False) & \
            (np.isnan(df_pht_140['mag'].values) == False) & \
            (np.isnan(df_pht_435['mag'].values) == False) & \
            (np.isnan(df_pht_606['mag'].values) == False) & \
            (np.isnan(df_pht_814['mag'].values) == False) & \
            (df_pht_110['mag'] < 30.) & (df_pht_140['mag'] < 30.) & \
            (df_pht_435['mag'] < 30.) & (df_pht_606['mag'] < 30.) & (df_pht_814['mag'] < 30.) & \
            (df_pht_435['mag']-df_pht_606['mag'] > -1.0) & \
            (df_pht_435['mag']-df_pht_606['mag'] < 2.0) & \
            (df_pht_606['mag']-df_pht_814['mag'] > -0.5) & \
            (df_pht_606['mag']-df_pht_814['mag'] < 1.5))


# ----- Figure setting: CMD ----- #
fig = plt.figure(1, figsize=(5.5,6))
gs = GridSpec(1, 1, left=0.20, bottom=0.15, right=0.95, top=0.95)

ax1 = fig.add_subplot(gs[0,0])

# Axis 1
ax = ax1
ax.set_xticks([-0.5,0.0,0.5,1.0,1.5])
ax.set_xticklabels([r'$-0.5$',0.0,0.5,1.0,1.5], fontsize=15.0)
ax.set_yticks([-24.0,-22.0,-20.0,-18.0,-16.0,-14.0,-12.0])
ax.set_yticklabels([r'$-24.0$',r'$-22.0$',r'$-20.0$',r'$-18.0$',
                    r'$-16.0$',r'$-14.0$',r'$-12.0$'], fontsize=15.0)
ax.set_xlabel(r'(F606W$-$F814W)$_{0}$', fontsize=15.0)
ax.set_ylabel(r'$M_{\rm F814W,0}$', fontsize=15.0)
ax.set_xlim([-0.5,1.5])
ax.set_ylim([-11.0,-24.0])
ax.tick_params(width=1.5, length=8.0)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))
ax.tick_params(width=1.5,length=5.0,which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)

X = d_sep2_606['mag_auto'].values - d_sep2_814['mag_auto'].values
Y = d_sep2_814['mag_auto'].values - dist_mod
X = X-(A_F606W-A_F814W)
Y = Y-A_F814W
ax.plot(X[plcmd], Y[plcmd], 'o', ms=3.0, color='gray', alpha=0.5)

idx_memb = np.loadtxt('sep_constraint.dat', dtype='int')
ax.plot(X[plcmd][idx_memb], Y[plcmd][idx_memb],
        'o', ms=4.0, color='red', alpha=0.9)

X = df_pht_606['mag'].values[knot_cnd] - df_pht_814['mag'].values[knot_cnd]
Y = df_pht_814['mag'].values[knot_cnd] - dist_mod
e_X = np.sqrt(df_pht_606['merr'].values[knot_cnd]**2. + df_pht_814['merr'].values[knot_cnd]**2.)
e_Y = df_pht_814['merr'].values[knot_cnd]
X = X-(A_F606W-A_F814W)
Y = Y-A_F814W
ax.errorbar(X, Y, xerr=e_X, yerr=e_Y,
            fmt='o', ms=7.5, mew=1.0, mfc='dodgerblue', mec='black', ecolor='dodgerblue',
            capsize=0, capthick=1.75, elinewidth=1.75, alpha=0.9)
ax.errorbar(d_sep2_606['mag_auto'].values[id_JFG2]-d_sep2_814['mag_auto'].values[id_JFG2]-(A_F606W-A_F814W),
            d_sep2_814['mag_auto'].values[id_JFG2]-dist_mod-A_F814W,
            fmt='*', ms=20.0, mew=1.5, mfc='orange', mec='black', ecolor='orange',
            capsize=0, capthick=2.0, elinewidth=2.0, alpha=1.0)

plt.savefig('fig_cmd.pdf')
plt.savefig('fig_cmd.png', dpi=300)
plt.close()


# ----- Figure setting: CCD ----- #
fig = plt.figure(2, figsize=(12,6))
gs = GridSpec(1, 2, left=0.10, bottom=0.15, right=0.98, top=0.96,
              width_ratios=[1.,1.], wspace=0.25)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

# Axis 1
ax = ax1
ax.set_xticks([-0.5,0.0,0.5,1.0,1.5,2.0])
ax.set_xticklabels([r'$-0.5$',0.0,0.5,1.0,1.5,2.0], fontsize=15.0)
ax.set_yticks([-1.0,-0.5,0.0,0.5,1.0,1.5])
ax.set_yticklabels([r'$-1.0$',r'$-0.5$',0.0,0.5,1.0,1.5], fontsize=15.0)
ax.set_xlabel(r'(F435W$-$F606W)$_{0}$', fontsize=15.0)
ax.set_ylabel(r'(F606W$-$F814W)$_{0}$', fontsize=15.0)
ax.set_xlim([-0.7,2.3])
ax.set_ylim([-0.8,1.2])
ax.tick_params(width=1.5, length=8.0)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
ax.tick_params(width=1.5,length=5.0,which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)

X = df_pht_435['mag'].values[knot_cnd] - df_pht_606['mag'].values[knot_cnd]
Y = df_pht_606['mag'].values[knot_cnd] - df_pht_814['mag'].values[knot_cnd]
e_X = np.sqrt(df_pht_435['merr'].values[knot_cnd]**2. + df_pht_606['merr'].values[knot_cnd]**2.)
e_Y = np.sqrt(df_pht_606['merr'].values[knot_cnd]**2. + df_pht_814['merr'].values[knot_cnd]**2.)
X = X-(A_F435W-A_F606W)
Y = Y-(A_F606W-A_F814W)
ax.errorbar(X, Y, xerr=e_X, yerr=e_Y,
            fmt='o', ms=7.5, mew=1.0, mfc='dodgerblue', mec='black', ecolor='dodgerblue',
            capsize=0, capthick=1.75, elinewidth=1.75, alpha=0.8)
ax.errorbar(d_sep2_435['mag_auto'].values[id_JFG2]-d_sep2_606['mag_auto'].values[id_JFG2]-(A_F435W-A_F606W),
            d_sep2_606['mag_auto'].values[id_JFG2]-d_sep2_814['mag_auto'].values[id_JFG2]-(A_F606W-A_F814W),
            fmt='*', ms=20.0, mew=1.5, mfc='orange', mec='black', ecolor='orange',
            capsize=0, capthick=2.0, elinewidth=2.0, alpha=0.9)


# ----- Loading the FSPS data ----- #
log_Z = np.array([0.0])
name_Z = ['m62']
name_sp = ['ssp0', 'tau0']
cs = ['red', 'green']
ls = ['-', '-']
sym = ['o', 's']
lb = ['Single burst SFH', 'Continuous SFH']
age = ini.age

# Plotting FSPS model
sp_mag = np.load('fsps_models/HSTACS_MIST_test.npz')
age_idx = np.array([0, 4, 9, 14, 19, 22, 23, 25, 26, 28, 29])
# age : 1 Myr, 5 Myr, 10 Myr, 15 Myr, 20 Myr, 50 Myr, 100 Myr, 500 Myr, 1 Gyr, 5 Gyr, 10 Gyr
for i in np.arange(len(name_sp)):
    name = name_sp[i]+'_m62'
    X = sp_mag[name][:,0,0] - sp_mag[name][:,0,1]
    Y = sp_mag[name][:,0,1] - sp_mag[name][:,0,2]
    ax.plot(X, Y, ls[i], color=cs[i], linewidth=1.75, alpha=0.6)
    p, = ax.plot(X[age_idx], Y[age_idx], sym[i], color=cs[i], ms=5.5, alpha=0.6, label=lb[i])
    exec("sym_"+name_sp[i]+" = p")

ax.legend(handles=[sym_ssp0, sym_tau0],
          fontsize=12.0, loc=(0.60,0.05), handlelength=0.0,
          numpoints=1, labelspacing=0.6, frameon=True, borderpad=0.6,
          framealpha=0.8, edgecolor='gray')

# Axis 2
ax = ax2
ax.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0])
ax.set_xticklabels([r'$-1.5$',r'$-1.0$',r'$-0.5$',0.0,0.5,1.0,1.5,2.0], fontsize=15.0)
ax.set_yticks([-2.0,-1.5, -1.0,-0.5,0.0,0.5,1.0,1.5])
ax.set_yticklabels([r'$-2.0$',r'$-1.5$',r'$-1.0$',r'$-0.5$',0.0,0.5,1.0,1.5], fontsize=15.0)
ax.set_xlabel(r'(F606W$-$F110W)$_{0}$', fontsize=15.0)
ax.set_ylabel(r'(F814W$-$F140W)$_{0}$', fontsize=15.0)
ax.set_xlim([-1.5,2.0])
ax.set_ylim([-1.75,1.5])
ax.tick_params(width=1.5, length=8.0)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
ax.tick_params(width=1.5,length=5.0,which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)

X = df_pht_606['mag'].values[knot_cnd] - df_pht_110['mag'].values[knot_cnd]
Y = df_pht_814['mag'].values[knot_cnd] - df_pht_140['mag'].values[knot_cnd]
e_X = np.sqrt(df_pht_606['merr'].values[knot_cnd]**2. + df_pht_110['merr'].values[knot_cnd]**2.)
e_Y = np.sqrt(df_pht_814['merr'].values[knot_cnd]**2. + df_pht_140['merr'].values[knot_cnd]**2.)
X = X-(A_F606W-A_F110W)
Y = Y-(A_F814W-A_F140W)
ax.errorbar(X, Y, xerr=e_X, yerr=e_Y,
            fmt='o', ms=7.5, mew=1.0, mfc='dodgerblue', mec='black', ecolor='dodgerblue',
            capsize=0, capthick=1.75, elinewidth=1.75, alpha=0.8)
ax.errorbar(d_sep2_606['mag_auto'].values[id_JFG2]-d_sep2_110['mag_auto'].values[id_JFG2]-(A_F606W-A_F110W),
            d_sep2_814['mag_auto'].values[id_JFG2]-d_sep2_140['mag_auto'].values[id_JFG2]-(A_F814W-A_F140W),
            fmt='*', ms=20.0, mew=1.5, mfc='orange', mec='black', ecolor='orange',
            capsize=0, capthick=2.0, elinewidth=2.0, alpha=0.9)

for i in np.arange(len(name_sp)):
    name = name_sp[i]+'_m62'
    X = sp_mag[name][:,0,1] - sp_mag[name][:,0,3]
    Y = sp_mag[name][:,0,2] - sp_mag[name][:,0,4]
    ax.plot(X, Y, ls[i], color=cs[i], linewidth=1.75, alpha=0.6)
    p, = ax.plot(X[age_idx], Y[age_idx], sym[i], color=cs[i], ms=5.5, alpha=0.6, label=lb[i])

plt.savefig('fig_ccd.pdf')
plt.savefig('fig_ccd.png', dpi=300)
plt.close()


f = open("apr3.reg", "w")
f.write("# Region file format: DS9 version 4.1\n")
f.write('global color=cyan dashlist=8 3 width=2 font="helvetica 10 normal roman"')
f.write('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
f.write("image\n")
for i in np.arange(np.sum(knot_cnd)):
    x = df_pht_435['x'].values[knot_cnd][i]
    y = df_pht_435['y'].values[knot_cnd][i]
    r = df_pht_435['aperture_rad'].values[knot_cnd][i]
    f.write("circle({0:.3f},{1:.3f},{2:.2f})\n".format(x,y,r))
f.close()
# os.system('ds9 -scale limits -0.01 0.05 -scale lock yes -frame lock image '+ \
#           '435_ori.fits 606_ori.fits 814_ori.fits sub.fits '+ \
#           '-tile grid mode manual -tile grid layout 4 1 &')



# ----- Figure setting: CMD ----- #
fig = plt.figure(3, figsize=(11,6))
gs = GridSpec(1, 2, left=0.10, bottom=0.15, right=0.98, top=0.95,
              width_ratios=[1.,1.], wspace=0.15)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

# Axis
naxis = 2
model_name = ['tau0_m62','ssp0_m62']
ls = ['-','--']
sym = ['s','o']
Ms = [1.0e+5, 1.0e+6, 1.0e+7, 1.0e+8, 1.0e+9, 1.0e+10, 1.0e+11, 1.0e+12]
obj_color = d_sep2_606['mag_auto'].values[id_JFG2]-d_sep2_814['mag_auto'].values[id_JFG2]-(A_F606W-A_F814W)
obj_Mag = d_sep2_814['mag_auto'].values[id_JFG2]-dist_mod-A_F814W

for nax in np.arange(naxis):
    exec(f"ax = ax{nax+1:d}")
    ax.set_xticks([-0.5,0.0,0.5,1.0,1.5])
    ax.set_xticklabels([r'$-0.5$',0.0,0.5,1.0,1.5], fontsize=15.0)
    ax.set_yticks([-24.0,-22.0,-20.0,-18.0,-16.0,-14.0,-12.0])
    ax.set_xlabel(r'(F606W$-$F814W)$_{0}$', fontsize=15.0)
    if (nax == 0):
        ax.set_yticklabels([r'$-24.0$',r'$-22.0$',r'$-20.0$',r'$-18.0$',
                            r'$-16.0$',r'$-14.0$',r'$-12.0$'], fontsize=15.0)
        ax.set_ylabel(r'$M_{\rm F814W,0}$', fontsize=15.0)
    else:
        ax.tick_params(labelleft=False)
    ax.set_xlim([-0.5,1.5])
    ax.set_ylim([-11.0,-24.0])
    ax.tick_params(width=1.5, length=8.0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=4))
    ax.tick_params(width=1.5,length=5.0,which='minor')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    X = d_sep2_606['mag_auto'].values - d_sep2_814['mag_auto'].values
    Y = d_sep2_814['mag_auto'].values - dist_mod
    X = X-(A_F606W-A_F814W)
    Y = Y-A_F814W
    ax.plot(X[plcmd], Y[plcmd], 'o', ms=3.0, color='gray', alpha=0.5)

    X = df_pht_606['mag'].values[knot_cnd] - df_pht_814['mag'].values[knot_cnd]
    Y = df_pht_814['mag'].values[knot_cnd] - dist_mod
    e_X = np.sqrt(df_pht_606['merr'].values[knot_cnd]**2. + df_pht_814['merr'].values[knot_cnd]**2.)
    e_Y = df_pht_814['merr'].values[knot_cnd]
    X = X-(A_F606W-A_F814W)
    Y = Y-A_F814W
    ax.errorbar(X, Y, xerr=e_X, yerr=e_Y,
                fmt='o', ms=7.5, mew=1.0, mfc='dodgerblue', mec='black', ecolor='dodgerblue',
                capsize=0, capthick=1.75, elinewidth=1.75, alpha=0.9)
    ax.errorbar(d_sep2_606['mag_auto'].values[id_JFG2]-d_sep2_814['mag_auto'].values[id_JFG2]-(A_F606W-A_F814W),
                d_sep2_814['mag_auto'].values[id_JFG2]-dist_mod-A_F814W,
                fmt='*', ms=20.0, mew=1.5, mfc='orange', mec='black', ecolor='orange',
                capsize=0, capthick=2.0, elinewidth=2.0, alpha=1.0)

    # Plotting models
    X = sp_mag[model_name[nax]][:,0,1] - sp_mag[model_name[nax]][:,0,2]
    Y = sp_mag[model_name[nax]][:,0,2] - dist_mod
    Ms0 = sp_mag[model_name[nax]][:,0,-1]
    for j in np.arange(len(Ms)):
        Y_copy = copy.deepcopy(Y)
        Y_copy -= 2.5*np.log10(Ms[j]/Ms0)
        ax.plot(X, Y_copy, ls[nax], color=f"C{j:d}", linewidth=1.75, alpha=0.6)
        p, = ax.plot(X[age_idx], Y_copy[age_idx], sym[nax], color=f"C{j:d}", ms=5.5, alpha=0.6)

    # Computing stellar mass
    dx = np.abs(X - obj_color)
    idx_color = dx.argmin()
    logMs = np.linspace(8., 12., 400+1)
    for m in np.arange(len(logMs)):
        Y_copy = copy.deepcopy(Y)
        Mag0 = Y_copy[idx_color] - 2.5*(logMs[m] - np.log10(Ms0[idx_color]))
        Mag1 = Y_copy[idx_color] - 2.5*(logMs[m+1] - np.log10(Ms0[idx_color]))
        if ((Mag0-obj_Mag)*(Mag1-obj_Mag) < 0.):
            obj_Ms = 10.0 ** (0.5*(logMs[m]+logMs[m+1]))
            break
    print("Galaxy age ("+model_name[nax]+f") : {age[idx_color]:.2f} Gyr")
    print("Galaxy stellar mass ("+model_name[nax]+f") : {obj_Ms:.3e} Mo")


plt.savefig('fig_cmd_Ms.pdf')
plt.savefig('fig_cmd_Ms.png', dpi=300)
plt.close()


# Computing stellar mass simply
obj_Mag_ = np.array([#d_sep2_435['mag_auto'].values[id_JFG2]-dist_mod-A_F435W,
                     d_sep2_606['mag_auto'].values[id_JFG2]-dist_mod-A_F606W,
                     d_sep2_814['mag_auto'].values[id_JFG2]-dist_mod-A_F814W])
for index_age in [-1, -2, -3, -4]:    # Age: 10, 5, 2, 1 Gyr
    ssp_Mag_ = sp_mag['ssp0_m62'][index_age,0,1:3] - dist_mod
    Ms_arr = sp_mag['ssp0_m62'][index_age,0,-1] * 2.5**(ssp_Mag_ - obj_Mag_)
    print(f"SSP model - Age: {age[index_age]:.1f} Gyr, Stellar mass: {np.mean(Ms_arr):.3e} Mo")



# Printing the running time
print('--- %s seconds ---' %(time.time()-start_time))
