#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 11:28:59 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os, copy
import fsps
# sp.libraries = (b'mist', b'miles')
import init_hstacs_mist_test as ini


# ----- Obtaining magnitudes ----- #
for i in np.arange(len(ini.name_sp)):
	for j in np.arange(len(ini.name_Z)):
		sp_mag = np.zeros((ini.n_age, ini.n_z, ini.n_band+1))
		exec("sp = ini.sp_"+ini.name_sp[i]+"_"+ini.name_Z[j])
		for k in np.arange(ini.n_age):
			for l in np.arange(ini.n_z):
				sp_mags = sp.get_mags(tage=ini.age[k], redshift=ini.z[l], bands=ini.acs_bands)
				sp_Ms = sp.stellar_mass
				sp_mag[k,l,:] = np.append(sp_mags, sp_Ms)
				# sp_mag[k,l,:] = sp.get_mags(tage=ini.age[k], redshift=ini.z[l], bands=ini.acs_bands)
		exec("sp_mag_"+ini.name_sp[i]+"_"+ini.name_Z[j]+" = copy.deepcopy(sp_mag)")


# ----- Saving arrays ----- #
os.system('rm -rfv '+ini.sav_name)
# np.savez_compressed(ini.sav_name,
# 	                ssp0_m42=sp_mag_ssp0_m42, ssp0_m62=sp_mag_ssp0_m62,
# 	                ssp1_m42=sp_mag_ssp1_m42, ssp1_m62=sp_mag_ssp1_m62,
# 	                tau0_m42=sp_mag_tau0_m42, tau0_m62=sp_mag_tau0_m62,
# 	                tau1_m42=sp_mag_tau1_m42, tau1_m62=sp_mag_tau1_m62)
np.savez_compressed(ini.sav_name,
	                ssp0_m62=sp_mag_ssp0_m62,
	                tau0_m62=sp_mag_tau0_m62)

# Printing the running time
print('--- %s seconds ---' %(time.time()-start_time))