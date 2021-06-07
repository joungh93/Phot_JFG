# Phot_JFG

@ Jeong Hwan Lee

# ----- Prerequisites ----- #
config.sex
output.param


# ----- Source codes ----- #
- mk_cutout.py
	# objname+"_"+filt+".fits" (cutout image)

- crmsk.py
	# "m"+objname+"_"+filt+".fits" (cr masked cutout image)

- mk_fltimg.py
	# *_ori.fits (cr masked cutout image), com.fits, sub.fits, apr.fits, knot.cat, apr.reg

(Saving knot2.reg (XY format) manually with DS9)

- apr_check1.py
	# knot2.reg (XY format) ---> apr2.reg (WCS format), apr2.txt

- apph2.py

- mkscr_sep2.py

- plt_sep2.py

- fsps_models/init_hstacs_mist_test.py
- fsps_models/set_hstacs_mist_test.py

- plt_ccd.py
	# apr3.reg (after photometry, removing )

- plt_map.py
