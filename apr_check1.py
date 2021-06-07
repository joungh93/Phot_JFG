#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:02:53 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
from astropy.io import fits
from matplotlib import pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy import units as u


# ----- Reading region files ----- #

# SExtractor output file
dat_knot = np.genfromtxt("knot.cat", dtype=None, encoding="ascii", comments="#",
                         usecols=(0,1,2,5,6,7),
                         names=('x','y','num','flag','fwhm','flxrad'))

# Manually selected knots
knot2 = np.genfromtxt("knot2.reg", dtype=None, encoding="ascii", names=('x','y'))


# ----- Matching & Writing region files ----- #
SRC = SkyCoord(x=dat_knot['x'], y=dat_knot['y'], z=0.,
	           unit='m', representation_type='cartesian')
KNT = SkyCoord(x=knot2['x'], y=knot2['y'], z=0.,
	           unit='m', representation_type='cartesian')
idx_SRC, d2d, d3d = KNT.match_to_catalog_sky(SRC)
tol = 0.01
matched= d3d.value < tol
idx_KNT = np.arange(len(KNT))[matched]

f = open("apr2.reg", "w")
g = open("apr2.txt", "w")
f.write("# Region file format: DS9 version 4.1\n")
f.write('global color=magenta dashlist=8 3 width=2 font="helvetica 10 normal roman"')
f.write('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
f.write("image\n")
for i in np.arange(len(matched)):
    x = dat_knot['x'][idx_SRC][i]
    y = dat_knot['y'][idx_SRC][i]
    r = dat_knot['flxrad'][idx_SRC][i]
    f.write("circle({0:.3f},{1:.3f},{2:.2f})\n".format(x,y,r))
    g.write("{0:.3f}  {1:.3f}  {2:.2f}\n".format(x,y,r))
f.close()
g.close()


# Printing the running time  
print("--- %s seconds ---" % (time.time() - start_time))