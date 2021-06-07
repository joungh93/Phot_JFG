#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:28:59 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import fsps
# sp.libraries = (b'mist', b'miles')


# ----- Directory for saving ----- #
sav_name = "HSTACS_MIST_test.npz"


# ----- Metallicity (GALAXEV) ----- #
# log_Z = np.array([-0.7, 0.0])
# name_Z = ['m42', 'm62']
# name_sp = ['ssp0', 'ssp1', 'tau0', 'tau1']
log_Z = np.array([0.0])
name_Z = ['m62']
name_sp = ['ssp0', 'tau0']


# ----- Creating stellar populations ----- #
for i in np.arange(len(name_sp)):
	for j in np.arange(len(name_Z)):

		# SSP model w/o nebular emission
		if (name_sp[i] == 'ssp0'):
			sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=1,
			                            sfh=0, logzsol=log_Z[j])

		# # SSP model w/ nebular emission
		# if (name_sp[i] == 'ssp1'):
		# 	sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=1,
		# 		                        sfh=0, logzsol=log_Z[j], gas_logz=log_Z[i], add_neb_emission=1)

		# Constant SFH w/o nebular emission
		if (name_sp[i] == 'tau0'):
			sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=1,
				                        sfh=1, logzsol=log_Z[j],
				                        tau=1.0, const=1.0, sf_start=0.0, sf_trunc=0.0, fburst=0.0, tburst=15.0)

		# # Constant SFH w/ nebular emission
		# if (name_sp[i] == 'tau1'):
		# 	sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1, imf_type=1,
		# 		                        sfh=1, logzsol=log_Z[j], gas_logz=log_Z[j], add_neb_emission=1,
		# 		                        tau=1.0, const=1.0, sf_start=0.0, sf_trunc=0.0, fburst=0.0, tburst=15.0)
	
		exec("sp_"+name_sp[i]+"_"+name_Z[j]+" = sp")


# ----- HST/ACS bands ----- #
acs_bands = ['wfc_acs_f435w', 'wfc_acs_f606w', 'wfc_acs_f814w',
             'wfc3_ir_f110w', 'wfc3_ir_f140w']
n_band = len(acs_bands)


# ----- Ages & redshifts ----- #
age = np.array([0.001, 0.002, 0.003, 0.004, 0.005,
	            0.006, 0.007, 0.008, 0.009, 0.010,
	            0.011, 0.012, 0.013, 0.014, 0.015,
	            0.016, 0.017, 0.018, 0.019, 0.020,
	            0.030, 0.040, 0.050,
	            0.100, 0.200, 0.500,
	            1., 2., 5., 10.])
n_age = len(age)

z = np.array([0.353])
n_z = len(z)

