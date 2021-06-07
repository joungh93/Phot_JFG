#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 8 13:51:43 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
from astropy.io import fits
from scipy.ndimage.filters import median_filter
import sep


# ----- Make an image list ----- #
diI = "./"
objname = "JFG2"
mi = glob.glob(diI+"m"+objname+"*.fits")
mi = sorted(mi)
acs_bands = ['435', '606', '814']


# ----- Create median filtered images ----- #
img = fits.getdata(mi[0])
dat = np.zeros((img.shape[0], img.shape[1], len(acs_bands)))
com = np.zeros((img.shape[0], img.shape[1], len(acs_bands)))

j = 0
for i in np.arange(len(mi)):
	mi_name = mi[i].split(diI)[1]
	filt = mi_name.split('.fits')[0].split('_')[1]
	os.system('cp -rpv '+mi[i]+' '+filt+'_ori.fits')
	# os.system('rm -rfv '+objname+'_'+filt+'.fits')
	# os.system('rm -rfv m'+objname+'_'+filt+'.fits')
	img, hdr = fits.getdata(filt+'_ori.fits', header=True)

	if (filt in acs_bands):
		mflt_img = median_filter(img, size=(10,10))
		com[:,:,j] = img
		dat[:,:,j] = img - mflt_img
		j += 1

fits.writeto('com.fits', np.sum(com, axis=2), hdr, overwrite=True)
fits.writeto('sub.fits', np.sum(dat, axis=2), hdr, overwrite=True)


# ----- Running SExtractor ----- #
f = open("knot.param","w")
for param in ['X_IMAGE', 'Y_IMAGE', 'NUMBER',
              'MAG_AUTO', 'MAGERR_AUTO',
              'FLAGS', 'FWHM_IMAGE', 'FLUX_RADIUS']:
    f.write(param+"\n")
f.close()

scr_sep = ''
# scr_sep += 'sex sub.fits,com.fits -c config.sex -CATALOG_NAME knot.cat '
scr_sep += 'sex sub.fits,com.fits -c config.sex -CATALOG_NAME knot.cat '
scr_sep += '-PARAMETERS_NAME knot.param '
scr_sep += '-DETECT_THRESH 1.5 -ANALYSIS_THRESH 1.5 '
scr_sep += '-DETECT_MINAREA 4 -DEBLEND_NTHRESH 32 -DEBLEND_MINCONT 0.001 '
scr_sep += '-MAG_ZEROPOINT 25.0 -GAIN 0.0 '
scr_sep += '-CHECKIMAGE_TYPE APERTURES -CHECKIMAGE_NAME apr.fits'
os.system(scr_sep)

os.system('ds9 -scalemode zscale -scale lock yes -frame lock image '+ \
	      ' 435_ori.fits 606_ori.fits 814_ori.fits '+ \
	      ' com.fits sub.fits '+ \
	      '-tile grid mode manual -tile grid layout 3 2 &')


# ----- Writing region files with aperture information ----- #
dat_knot = np.genfromtxt("knot.cat", dtype=None, encoding="ascii", comments="#",
                         usecols=(0,1,2,5,6,7),
                         names=('x','y','num','flag','fwhm','flxrad'))

f = open("apr.reg", "w")
f.write("# Region file format: DS9 version 4.1\n")
f.write('global color=yellow dashlist=8 3 width=2 font="helvetica 10 normal roman"')
f.write('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
f.write("image\n")
for i in np.arange(len(dat_knot)):
	x = dat_knot['x'][i]
	y = dat_knot['y'][i]
	r = dat_knot['flxrad'][i]
	f.write(f"circle({x:.3f},{y:.3f},{r:.2f})\n")
f.close()


# Printing the running time  
print("--- %s seconds ---" % (time.time() - start_time))