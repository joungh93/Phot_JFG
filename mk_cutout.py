#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:53:57 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
import pandas as pd
import copy
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy import wcs


# ----- Initial settings ----- #
dir_Img = "/data/jlee/DATA/HLA/McPartland+16/MACS1752/Img_total/HST_init/"
objname = "JFG2"
imgname = glob.glob(dir_Img+"calw*.fits")
imgname = sorted(imgname)


# ----- Making cutout images ----- #
for i in np.arange(len(imgname)):
	filt = imgname[i].split('/')[-1].split('calw')[1].split('.fits')[0]
	dat, hdr = fits.getdata(imgname[i], header=True, ext=0)
	w = wcs.WCS(hdr)

	chdu = Cutout2D(data=dat, position=(1270, 4220), size=(400, 400), wcs=w)
	h = chdu.wcs.to_header()
	
	h['EXPTIME'] = hdr['EXPTIME']
	h['MAGZERO'] = -2.5*np.log10(hdr['PHOTFLAM'])-5.*np.log10(hdr['PHOTPLAM'])-2.408
	h['CCDGAIN'] = hdr['CCDGAIN']

	fits.writeto(objname+"_"+filt+".fits", chdu.data, h, overwrite=True)


# Printing the running time  
print("--- %s seconds ---" % (time.time() - start_time))
