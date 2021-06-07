#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 18:46:58 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from astropy.io import fits
from astropy import wcs
from astropy.cosmology import FlatLambdaCDM
import imgscl


# ----- Directories ----- #
diH = '/data/jlee/DATA/HLA/McPartland+16/MACS1752/JFG2/Phot/'
diG = '/data/jlee/DATA/Gemini/Programs/GN-2019A-Q-215/redux4_700/'


# ----- Basic parameters ----- #
redshift = 0.3527  # McPartland+16 Table 1
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
lum_dist = cosmo.luminosity_distance(redshift).value  # Mpc
dist_scale = 1. / cosmo.arcsec_per_kpc_proper(redshift).value  # kpc/arcsec
pixel_scale = 0.05  # pixel/arcsec
ifu_w = 7.0 / pixel_scale  # pixel
ifu_h = 5.0 / pixel_scale  # pixel
PA, ang0 = 85.0, 0.02957  # arcdeg
galname = "MACSJ1752-JFG2"


# ----- Reading FITS files ----- #
# img435 = fits.getdata(diH+'435_ori.fits', ext=0)
# img606 = fits.getdata(diH+'606_ori.fits', ext=0)
# img814 = fits.getdata(diH+'814_ori.fits', ext=0)
img435 = fits.getdata('435_ori.fits', ext=0)
img606 = fits.getdata('606_ori.fits', ext=0)
img814 = fits.getdata('814_ori.fits', ext=0)
imgsub = fits.getdata('sub.fits', ext=0)

hdr1 = fits.getheader(diG+'cstxeqxbrgN20190611S0257_3D.fits', ext=0)
gra = hdr1['RA'] ; gdec = hdr1['DEC']


# ----- Creating RGB data ----- #
cimg = np.zeros((img814.shape[0], img814.shape[1], 3), dtype=float)
cimg[:,:,0] = imgscl.linear(0.5*img814, scale_min=-0.02, scale_max=0.075)   # R
cimg[:,:,1] = imgscl.linear(0.5*(img606+img814), scale_min=-0.02, scale_max=0.15)   # G
cimg[:,:,2] = imgscl.linear(0.5*img606, scale_min=-0.02, scale_max=0.075)   # B


# ----- WCS to XY ----- #
# h = fits.getheader(diH+'606_ori.fits', ext=0)
h = fits.getheader('606_ori.fits', ext=0)
w = wcs.WCS(h)
px, py = w.wcs_world2pix(gra, gdec, 1)
rth = 115
isz = 2*rth


# ----- Figure setting ----- #
fig = plt.figure(1, figsize=(12,6))
gs = GridSpec(1, 2, left=0.025, bottom=0.05, right=0.975, top=0.95,
              width_ratios=[1.,1.], wspace=0.05)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])


# ----- Axis 1 ----- #
ax = ax1
ax.tick_params(labelleft=False)
ax.tick_params(labelbottom=False)
ax.tick_params(width=0.0, length=5.0)
ax.imshow(cimg[int(py-1-rth):int(py-1+rth),
               int(px-1-rth):int(px-1+rth),:],
          aspect='equal', origin='lower')

# IFU FOV
ang = (PA - ang0) * (np.pi/180.0) ; x_off = 0.0 ; y_off = 0.0
rot = np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])
spoint = np.array([[-0.5*ifu_w,+0.5*ifu_h],[+0.5*ifu_w,+0.5*ifu_h],
                   [+0.5*ifu_w,-0.5*ifu_h],[-0.5*ifu_w,-0.5*ifu_h]])
epoint = np.array([[+0.5*ifu_w,+0.5*ifu_h],[+0.5*ifu_w,-0.5*ifu_h],
                   [-0.5*ifu_w,-0.5*ifu_h],[-0.5*ifu_w,+0.5*ifu_h]])
for i in range(4):
    srot = np.dot(rot,spoint[i]) ; erot = np.dot(rot,epoint[i])
    ax.arrow(srot[0]+rth+x_off, srot[1]+rth+y_off, erot[0]-srot[0], erot[1]-srot[1], width=2.0,
             head_width=0., head_length=0., fc='cyan', ec='cyan', alpha=0.8)

# Scale bar
kpc10 = 10.0/dist_scale/pixel_scale
ax.arrow(0.05*isz, 0.10*isz, kpc10, 0., width=2.25, head_width=0., head_length=0.,
         fc='yellow', ec='yellow', alpha=0.8)
ax.text(0.05, 0.03, '10 kpc', fontsize=17.5, fontweight='bold', color='yellow',
	    ha='left', va='bottom', transform=ax.transAxes)

# The orientations
x0, y0 = 0.95*isz, 0.80*isz
L = 0.10*isz
theta0 = -ang0*(np.pi/180.0)
ax.arrow(x0, y0-1.0, -L*np.sin(theta0), L*np.cos(theta0), width=2.25,
         head_width=7.5, head_length=7.5, fc='yellow', ec='yellow', alpha=0.8)
ax.arrow(x0+1.0, y0, -L*np.cos(theta0), -L*np.sin(theta0), width=2.25,
         head_width=7.5, head_length=7.5, fc='yellow', ec='yellow', alpha=0.8)
ax.text(0.78, 0.78, 'E', fontsize=17.5, fontweight='bold', color='yellow',
	    ha='left', va='bottom', transform=ax.transAxes)
ax.text(0.93, 0.94, 'N', fontsize=17.5, fontweight='bold', color='yellow',
	    ha='left', va='bottom', transform=ax.transAxes)

# Target name
ax.text(0.04, 0.96, galname, fontsize=20.0, fontweight='bold', color='white',
	    ha='left', va='top', transform=ax.transAxes)
ax.text(0.06, 0.89, '(z=%.3f)' %(redshift), fontsize=18.0, fontweight='bold', color='white',
	    ha='left', va='top', transform=ax.transAxes)


# ----- Axis 2 ----- #
ax = ax2
ax.tick_params(labelleft=False)
ax.tick_params(labelbottom=False)
ax.tick_params(width=0.0, length=5.0)
ax.imshow(imgsub[int(py-1-rth):int(py-1+rth),
                 int(px-1-rth):int(px-1+rth)],
          origin='lower', cmap='gray_r', vmin=-0.01, vmax=0.05)

# Aperture map
apr = np.genfromtxt(diH+'apr3.reg', dtype=str, skip_header=3, encoding='ascii')
for i in np.arange(apr.size):
	x0 = float(apr[i].split(',')[0].split('(')[1])
	y0 = float(apr[i].split(',')[1])
	r0 = float(apr[i].split(',')[2].split(')')[0])

	c0 = plt.Circle((x0-int(px-rth), y0-int(py-rth)), r0,
		            color='magenta', linewidth=2.0, linestyle='-',
		            alpha=0.8, fill=False)
	ax.add_artist(c0)


plt.savefig('fig_map.png', dpi=300)
plt.savefig('fig_map.pdf')
plt.close()


# ----- Figure setting ----- #
fig = plt.figure(2, figsize=(6,6))
gs = GridSpec(1, 1, left=0.025, bottom=0.025, right=0.975, top=0.975)

ax1 = fig.add_subplot(gs[0,0])


# ----- Axis 1 ----- #
ax = ax1
ax.tick_params(labelleft=False)
ax.tick_params(labelbottom=False)
ax.tick_params(width=0.0, length=5.0)
ax.imshow(cimg[int(py-1-rth):int(py-1+rth),
               int(px-1-rth):int(px-1+rth),:],
          aspect='equal', origin='lower')

# Scale bar
kpc10 = 10.0/dist_scale/pixel_scale
ax.arrow(0.05*isz, 0.10*isz, kpc10, 0., width=2.25, head_width=0., head_length=0.,
         fc='yellow', ec='yellow', alpha=0.8)
ax.text(0.05, 0.03, '10 kpc', fontsize=17.5, fontweight='bold', color='yellow',
      ha='left', va='bottom', transform=ax.transAxes)

# The orientations
x0, y0 = 0.95*isz, 0.80*isz
L = 0.10*isz
theta0 = -ang0*(np.pi/180.0)
ax.arrow(x0, y0-1.0, -L*np.sin(theta0), L*np.cos(theta0), width=2.25,
         head_width=7.5, head_length=7.5, fc='yellow', ec='yellow', alpha=0.8)
ax.arrow(x0+1.0, y0, -L*np.cos(theta0), -L*np.sin(theta0), width=2.25,
         head_width=7.5, head_length=7.5, fc='yellow', ec='yellow', alpha=0.8)
ax.text(0.78, 0.78, 'E', fontsize=17.5, fontweight='bold', color='yellow',
      ha='left', va='bottom', transform=ax.transAxes)
ax.text(0.93, 0.94, 'N', fontsize=17.5, fontweight='bold', color='yellow',
      ha='left', va='bottom', transform=ax.transAxes)

# Target name
ax.text(0.04, 0.96, galname, fontsize=20.0, fontweight='bold', color='white',
      ha='left', va='top', transform=ax.transAxes)
ax.text(0.06, 0.89, '(z=%.3f)' %(redshift), fontsize=18.0, fontweight='bold', color='white',
      ha='left', va='top', transform=ax.transAxes)


plt.savefig('fig_map2.png', dpi=300)
plt.savefig('fig_map2.pdf')
plt.close()



# Printing the running time
print('--- %s seconds ---' %(time.time()-start_time))

