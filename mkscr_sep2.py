#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:45:24 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
from astropy.io import fits
import pandas as pd


# ----- Making a detection image ----- #
dir_Img = "/data/jlee/DATA/HLA/McPartland+16/MACS1752/Img_total/HST_init/"
calw = glob.glob(dir_Img+"calw*.fits")
calw = sorted(calw)
acs_bands = ['435', '606', '814']

dat = fits.getdata(calw[0], ext=0)
cdat = np.zeros((dat.shape[0], dat.shape[1], len(acs_bands)))
out = 'calw_dtc.fits'

j = 0
for i in np.arange(len(calw)):
	filt = calw[i].split(dir_Img)[1].split('calw')[1].split('.fits')[0]
	if (filt in acs_bands):
		dat, hdr = fits.getdata(calw[i], header=True, ext=0)
		cdat[:,:,j] = dat
		j += 1
cdat2 = np.mean(cdat, axis=2)
fits.writeto(out, cdat2, hdr, overwrite=True)
dtc = out


# ----- Parameter setting ----- #
DETECT_THRESH = 2.0
DETECT_MINAREA = 5
DEBLEND_NTHRESH = 32
DEBLEND_MINCONT = 0.05
PHOT_APERTURES = [3.0, 6.0, 8.0]
CHECKIMAGE_TYPE = ['APERTURES']
CHECKIMAGE_NAME = ['apr']
# ----------------------------- #


# ----- Writing the script ----- #
f = open('sep2.sh','w')
f.write(' \n')
f.write('##################################'+'\n')
f.write('##### Scripts for SExtractor #####'+'\n')
f.write('##################################'+'\n')        
f.write(' \n')
for i in np.arange(len(calw)):
	flt = calw[i].split(dir_Img)[1].split('calw')[1].split('.fits')[0]

	dat, hdr = fits.getdata(calw[i], header=True, ext=0)

	gain = hdr['CCDGAIN']*hdr['EXPTIME']
	zp = -2.5*np.log10(hdr['PHOTFLAM'])-5.0*np.log10(hdr['PHOTPLAM'])-2.408

	f.write('sex '+dtc+','+calw[i]+' -c config.sex -CATALOG_NAME sep2_'+flt+'.cat ')
	f.write('-DETECT_THRESH {0:.1f} -ANALYSIS_THRESH {0:.1f} '.format(DETECT_THRESH))
	f.write('-DETECT_MINAREA {0:.1f} -DEBLEND_NTHRESH {1:d} -DEBLEND_MINCONT {2:.3f} '.format(DETECT_MINAREA, DEBLEND_NTHRESH, DEBLEND_MINCONT))
	f.write('-MAG_ZEROPOINT {0:.3f} -GAIN {1:.1f} '.format(zp, gain))
	
	photap = ''
	for j in np.arange(len(PHOT_APERTURES)):
		if (j != np.arange(len(PHOT_APERTURES))[-1]):
			photap += '{0:.1f}'.format(PHOT_APERTURES[j])+','
		else:
			photap += '{0:.1f}'.format(PHOT_APERTURES[j])
	f.write('-PHOT_APERTURES '+photap+' ')

	check_type, check_name = '', ''
	for j in np.arange(len(CHECKIMAGE_TYPE)):
		if (j != np.arange(len(CHECKIMAGE_TYPE))[-1]):
			check_type += CHECKIMAGE_TYPE[j]+','
			check_name += flt+'_'+CHECKIMAGE_NAME[j]+'.fits,'
		else:
			check_type += CHECKIMAGE_TYPE[j]
			check_name += flt+'_'+CHECKIMAGE_NAME[j]+'.fits'
	f.write('-CHECKIMAGE_TYPE '+check_type+' -CHECKIMAGE_NAME '+check_name+' ')
	
	f.write('\n')
f.close()


# ----- Running scripts for SExtractor ----- #
os.system('sh sep2.sh')


# ----- Reading & saving the results ----- #
colname = ['x','y','num','mag_auto','merr_auto',
           'mag1','mag2','mag3','merr1','merr2','merr3',
           'kron','bgr','ra','dec','cxx','cyy','cxy',
           'a','b','theta','mu0','mut','flag','fwhm','flxrad','cl']

for i in np.arange(len(calw)):
	flt = calw[i].split(dir_Img)[1].split('calw')[1].split('.fits')[0]
	dat = np.genfromtxt('sep2_'+flt+'.cat', dtype=None, comments='#', encoding='ascii',
                        names=colname)
	dat = pd.DataFrame(dat)
	dat.to_pickle('d_sep2_'+flt+'.pkl')


# ----- Running DS9 ----- #
imglist = ''
for i in np.arange(len(calw)):
	imglist += calw[i]+' '
os.system('ds9 -scale limits -0.01 0.05 -scale lock yes -frame lock image '+imglist+ \
	      '814_apr.fits -tile grid mode manual -tile grid layout 6 1 &')


# Printing the running time
print('--- %s seconds ---' %(time.time()-start_time))