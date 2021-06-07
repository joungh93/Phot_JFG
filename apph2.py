#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:04:16 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
import pandas as pd
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
# from photutils import DAOStarFinder
from photutils.aperture import CircularAperture as CAp
from photutils.aperture import CircularAnnulus as CAn
from photutils import aperture_photometry as apphot
from tqdm import trange

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


# ----- Reading data ----- #
img_orig = glob.glob("*_ori.fits")
img_orig = sorted(img_orig)

knot2 = np.genfromtxt("apr2.txt", dtype=None, encoding="ascii",
                      names=('x','y','r'))


# ----- Running aperture photometry ----- #
cols = [('id','<i8'), ('x','<f8'), ('y','<f8'), ('aperture_rad','<f8'),
        ('aperture_sum','<f8'), ('area_ap','<f8'),
        ('msky','<f8'), ('nsky','<f8'), ('sky_sigma','<f8'),
        ('source_sum','<f8'), ('mag','<f8'), ('merr','<f8')]
# zmag = [25.667, 26.500, 25.946]
# gain = [2.0, 2.0, 2.0]
# itime = [1.0, 1.0, 1.0]
# exptime = [2526., 2534., 2394.] 

for j in np.arange(len(img_orig)):
	filt = img_orig[j].split('_ori.fits')[0]
	phot_table = np.zeros(len(knot2), dtype=cols)

	data, head = fits.getdata(img_orig[j], header=True, ext=0)
	zmag = head['MAGZERO']
	gain = head['CCDGAIN']
	exptime = head['EXPTIME']
	itime = 1.0

	for i in trange(len(knot2)):
		pos = [(knot2['x'][i], knot2['y'][i])]
		r_ap = knot2['r'][i]
		ap = CAp(positions=pos, r=r_ap)
		an = CAn(positions=pos, r_in=3*r_ap, r_out=5*r_ap)

		# Aperture photometry
		phot = apphot(data=data, apertures=ap)

		phot_table['id'][i] = i+1
		phot_table['x'][i] = phot['xcenter'].data[0]
		phot_table['y'][i] = phot['ycenter'].data[0]
		phot_table['aperture_rad'][i] = ap.r
		phot_table['aperture_sum'][i] = phot['aperture_sum'].data[0]
		phot_table['area_ap'][i] = ap.area

		# Local sky estimation
		sky_mask = an.to_mask(method='center')
		sky_vals = sky_mask[0].multiply(data)
		skyv = sky_vals[sky_vals != 0.]
		nsky = np.sum(sky_vals != 0.)
		avg, med, std = sigma_clipped_stats(skyv, sigma=3, maxiters=5, std_ddof=1)
		if med - avg < 0.3 * std:
		    msky = med
		else:
		    msky = 2.5 * med - 1.5 * avg

		# Magitude calculation
		src_sum = phot['aperture_sum'].data[0] - ap.area*msky
		nflx = src_sum / itime
		tflx = nflx * exptime
		mag = zmag - 2.5*np.log10(nflx)
		err = np.sqrt(tflx/gain + ap.area*std**2. + (ap.area**2. * std**2.)/nsky)

		phot_table['msky'][i] = msky
		phot_table['nsky'][i] = nsky
		phot_table['sky_sigma'][i] = std
		phot_table['source_sum'][i] = src_sum
		phot_table['mag'][i] = mag
		phot_table['merr'][i] = (2.5/np.log(10.0)) * (err/tflx)

	df_name = "df_pht_"+filt
	exec(df_name+" = pd.DataFrame(phot_table)")
	exec(df_name+".to_pickle('"+df_name+".pkl')")


# Printing the running time  
print("--- %s seconds ---" % (time.time() - start_time))