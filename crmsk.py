#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:22:14 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
import pandas as pd
import copy
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.nddata import Cutout2D
from astropy import wcs
from astropy.stats import sigma_clipped_stats
from tqdm import trange


# ----- Make an image list ----- #
objname = "JFG2"
imgname = glob.glob(objname+"_*.fits")
imgname = sorted(imgname)


# ----- Circular masking ----- #
msk = [(256., 160., 3.),
       (273., 137., 5.),
       (223., 129., 4.),
       (260., 256., 3.),
       (305., 263., 4.),
       (180., 155., 3.),
       (228., 267., 4.),
       (274., 160., 2.),
       (289., 185., 3.),
       (292., 181., 3.),
       (279., 140., 3.),
       (230., 89., 3.),
       (221., 87., 3.),
       (248., 87., 3.),
       (292., 116., 3.),
       (305., 120., 3.),
       (241., 115., 3.)]
       # (x, y, r)
# umsk = (138, 107)  # in jfg

kernel = Gaussian2DKernel(5)

for i in np.arange(len(imgname)):
	filt = imgname[i].split(objname+"_")[1].split(".fits")[0]
	print("Masking "+imgname[i]+"...")

	dat, hdr = fits.getdata(imgname[i], header=True, ext=0)
	w = wcs.WCS(hdr)

	dat2 = copy.deepcopy(dat)
	fl_msk = np.zeros_like(dat)

	x = np.arange(0, dat.shape[1], 1)
	y = np.arange(0, dat.shape[0], 1)
	xx, yy = np.meshgrid(x, y, sparse=True)


	# Circular masking
	for j in trange(len(msk)):
		z = (xx-(msk[j][0]-1))**2.0 + (yy-(msk[j][1]-1))**2.0 - (1.0*msk[j][2])**2.0
		mskpix = (z <= 0.0)
		dat2[mskpix] = np.nan
		fl_msk[mskpix] = 1

	img_conv = convolve(dat2, kernel)
	dat2[fl_msk == 1] = img_conv[fl_msk == 1]


	# # Region masking
	# avg, med, std = sigma_clipped_stats(img2, sigma=3.0, maxiters=10)
	# np.random.seed(0)
	# img2[np.isnan(img2) == True] = np.random.normal(med, std, img2[np.isnan(img2) == True].shape)
	# img2[:20, :] = np.random.normal(med, std, img2[:20, :].shape)
	# img2[:60, 181:] = np.random.normal(med, std, img2[:60, 181:].shape)
	# img2[50:100, 250:300] = np.random.normal(med, std, img2[50:100, 250:300].shape)

	# Creating new images
	fits.writeto("m"+objname+"_"+filt+".fits", dat2, hdr, overwrite=True) 


# Printing the running time  
print("--- %s seconds ---" % (time.time() - start_time))