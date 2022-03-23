from __future__ import division
from __future__ import print_function

from builtins import zip
import os, sys, scipy, hashlib, glob, subprocess, imp, pdb, copy
import pickle as pk
import healpy as hp
import numpy as np
import pylab as pl
import datetime
import glob
from spt3g import core, maps
from spt3g.maps import FlatSkyMap
from spt3g.mapspectra import MapSpectrum1D
from spt3g.lensing.map_spec_utils import MapSpectraTEB
from spt3g import lensing as sl
import healpy

uK = core.G3Units.uK
arcmin = core.G3Units.arcmin

# Based on parameter file /home/panz/code/spt3g_software/lensing/specific_analyses/lens3g/params/par_run_150ghz_12.py
# and /home/panz/code/spt3g_software/lensing/specific_analyses/lens3g/params/par_run_150ghz_06.py.

#----------------------
# Cuts 
#----------------------
lmax_sinv    = 5400 # Cut off the filtered signal at this value

# When evalutating phi, cutoff the integral between 'lmax_cinv' and 'lx_cut'
lmax_cinv    = 5400 # Cut off the cinv-filtered fields at this value
lmin_cinv    = 1
lxmin_cinv   = 1

# T to P leakage
tq_leak = 0.00463
tu_leak = 0.01278

# Polarization calibration factor 
# Note that this is already corrected for the data or sims
pcal = 1.05

#----------------------
# Names 
#----------------------
ivfs_prefix = "run_01"
qest_prefix = "run_01"

# Specify which sims are used for which quadratic cl estimator (qecl)
nsim = 500
mc_sims = np.arange(0,nsim) # Indices for all sims
mc_sims_mf = mc_sims[0:160] # These are the sims used for mean field estimation
mc_sims_var = mc_sims[161:500] # Variance
mc_sims_unl = mc_sims_var # Unlensed sims
mc_sims_qcr_mc = mc_sims_var # The qcr, normalization calculation
mc_sims_n0 = mc_sims # N0 bias
mc_sims_n1 = mc_sims[1:100]#mc_sims[4:164] # N1 bias

# cmbs
print("---- get cmbs")
bdir = '/home/yukanakato/spt3g_software_tutorial/qest/run_01/'
lib_dir = bdir

res = 2*arcmin
ny = 390
nx = 390
# 3G full survey projected depth at 150 GHz has the below noise levels
nlev_t = 2.2*uK*arcmin # In G3 units
nlev_p = np.sqrt(2) * nlev_t

parent = FlatSkyMap(nx, ny, res,
                    alpha_center=0.0*core.G3Units.deg,
                    delta_center=-57.5*core.G3Units.deg,
                    proj=maps.MapProjection.ProjZEA,
                    coord_ref=maps.MapCoordReference.Equatorial,
                    pol_conv=maps.MapPolConv.IAU,)

# CMB only (no foregrounds), no filtering simulation maps (these projected maps are not beamed, i.e. bl is np.ones(lmax))
cmblib_len_t1p1 = sl.obs.SimLib(lib_dir=lib_dir+'sims_t1p1/', name_fmt='sim_{:n}.g3', parent=parent)
cmblib_len_t2p1_nofg = sl.obs.SimLib(lib_dir=lib_dir+'sims_t2p1/', name_fmt='sim_{:n}.g3', parent=parent)
cmblib_unl_t1p1 = sl.obs.SimLib(lib_dir=lib_dir+'sims_unl/', name_fmt='sim_{:n}.g3', parent=parent)
cmblib_kap = sl.obs.SimLib(lib_dir=lib_dir+'sims_kap/', parent=parent, name_fmt='sim_{:n}.g3')

# Theory spectra
lmax_theoryspec = 5400 # Cut off the theory spectrum at this value  
cl_unl = sl.map_spec_utils.get_camb_scalcl(prefix="planck18_TTEEEE_lowl_lowE_lensing_highacc", lmax=lmax_theoryspec)
cl_unl["BB"] = np.zeros(lmax_theoryspec+1)
cl_len = sl.map_spec_utils.get_camb_lensedcl(prefix="planck18_TTEEEE_lowl_lowE_lensing_highacc", lmax=lmax_theoryspec)

print("---- get beams")
parent_big = FlatSkyMap(int(75*core.G3Units.deg//res), int(50*core.G3Units.deg//res), res,
                    alpha_center=0.0*core.G3Units.deg,
                    delta_center=-57.5*core.G3Units.deg,
                    proj=maps.MapProjection.ProjZEA,
                    coord_ref=maps.MapCoordReference.Equatorial,
                    pol_conv=maps.MapPolConv.IAU,)
# Load up the beam
beam_ell, beam_bl = np.loadtxt('/sptlocal/user/panz/lensing_150ghz/inputs/mars_beam_2018.txt',
                               usecols=[0,2], unpack=True)
map_cal_factor = 1.0 # Maps were already calibrated on a subfield basis
bl = MapSpectrum1D(np.concatenate((beam_ell-0.5, [max(beam_ell)+0.5])), beam_bl)

# Convert the beam into an object in 2D Fourier space
# Fill up the 2D Fourier plane with the same value for all the annulus with same ell
tf_beam = (MapSpectraTEB(ffts=3*[np.ones([int(50*core.G3Units.deg//res), int(75*core.G3Units.deg//res)//2+1])],
                         parent=parent_big,) * bl).get_l_masked(lmax=lmax_theoryspec)
tf_ones = (MapSpectraTEB(ffts=3*[np.ones([ny, nx//2+1])],
                         parent=parent,))
tf_ones = tf_ones.get_l_masked(lmax=lmax_theoryspec) * tf_ones.get_pixel_window()

# Load up the 2D transfer function
print("---- get transfer function")
tf_file = '/sptlocal/user/panz/lensing_150ghz/inputs/tf_150GHz.pk'
tf = pk.load(open(tf_file, 'rb'))

# Convert the transfer function into an tebfft object in 2D Fourier space
tf = (MapSpectraTEB(ffts=[np.array(tf['T'].get_real()), np.array(tf['E'].get_real()), np.array(tf['B'].get_real())],
                    parent=parent_big) * bl * tf_beam.get_pixel_window()).get_l_masked(lmax=lmax_theoryspec)

print("---- data and sims")
# Lensed sims (signal and noise)
obs_len = sl.obs.ObsHomogeneousNoise(data=None,
                                     signal_lib=cmblib_len_t1p1, # Needs to be a SimLib object for it to work
                                     tq=tq_leak, tu=tu_leak,
                                     thresh=1000.0*uK,
                                     tcal=1.0, pcal=pcal,
                                     nlev_t=nlev_t,
                                     nlev_p=nlev_p)

# Unlensed sims
obs_unl = sl.obs.ObsHomogeneousNoise(data=None,
                                     signal_lib=cmblib_unl_t1p1,
                                     tq=tq_leak, tu=tu_leak,
                                     thresh=1000.0*uK,
                                     tcal=1.0, pcal=pcal)

# For N1 bias; nofg means no foreground; t2p1 means same lensing realizations but different CMB realizations
# No foreground is needed for N1 bias
obs_len_t2_nofg = sl.obs.ObsHomogeneousNoise(signal_lib=cmblib_len_t2p1_nofg,
                                             thresh=1000.0*uK,)
# For calculating the N1 bias as: len_t2 - len_nofg; t1p1 means different lensing realizations and different CMB realizations
obs_len_t1_nofg = sl.obs.ObsHomogeneousNoise(signal_lib=cmblib_len_t1p1,
                                             thresh=1000.0*uK,)

# ivfs
print("---- ivfs")
apod150 = np.load('/scratch/panz/lens100d_py3/inputs/mask/apod_surv_five_150.npy') # Used for plotting purposes only     
mask150 = np.load('/scratch/panz/lens100d_py3/inputs/mask/mask_surv_five_150.npy') * np.load('/scratch/panz/lens100d_py3/inputs/mask/mask_clust10.npy') * np.load('/scratch/panz/lens100d_py3/inputs/mask/mask_srccr10.npy')

print("---- ivfs:ninv")
ninv = sl.map_spec_utils.make_tqumap_wt(ninv=None,
                                        parent=parent,
                                        ninv_dcut=None,
                                        nlev_tp=(nlev_t, nlev_p),
                                        mask=apod150,)

# Map space noise 
#ninvfilt = sl.cinv_utils.opfilt_teb.NoiseInverseFilter(tf, ninv)
ninvfilt = sl.cinv_utils.opfilt_teb.NoiseInverseFilter(tf_ones, ninv)

print("---- ivfs:sinvfilt")
# Need to delete clte, otherwise the iteration for solving matrix inversion won't converge
cl_len_filt = {}
for k in cl_len.keys():
    cl_len_filt[k] = cl_len[k][0:lmax_sinv+1]
del cl_len_filt["TE"]  # Theoretical spectrum

# Includes signal and Fourier space noise
total_cl = {}
'''
total_cl['L'], total_cl['TT'], total_cl['EE'], total_cl['BB'] = np.loadtxt(bdir+'/inputs/powerspectrum_cl_cmb_and_foreground_for_2018_lensing.txt', usecols=[0,2,5,8], unpack=True)

for k in ['TT','EE','BB']:
    total_cl[k] = np.concatenate((np.zeros(int(min(total_cl['L']))), total_cl[k])) * uK**2
total_cl['L'] = np.concatenate((np.arange(min(total_cl['L'])), total_cl['L']))
'''
total_cl = cl_len_filt

nl2d = pk.load(open("/sptlocal/user/panz/lensing_150ghz/inputs/noise_150GHz.pk", "rb"))
'''
nl2d = pk.load(open("/sptlocal/user/panz/lensing_150ghz/inputs/noise_150GHz.pk", "rb"))
nl2d = MapSpectraTEB(ffts=[np.array(nl2d["TT"].get_real()),
                           np.array(nl2d["EE"].get_real()),
                           np.array(nl2d["BB"].get_real()),],
                           parent=parent,)
'''
# nl2d['TT'].get_real().shape is (1500, 1126), but we should crop to (390, 390)
nl2d = MapSpectraTEB(
    ffts=[
        np.array(np.zeros((390, 390//2+1))),
        np.array(np.zeros((390, 390//2+1))),
        np.array(np.zeros((390, 390//2+1))),
    ],
    parent=parent,
)

# Calculate S^{-1}
#sinvfilt = sl.cinv_utils.opfilt_teb.cl2sinv(total_cl, nl2d, tf, nft=nlev_t, nfp=nlev_p, lmax=lmax_sinv)
sinvfilt = sl.cinv_utils.opfilt_teb.cl2sinv(total_cl, nl2d, tf_ones, nft=nlev_t, nfp=nlev_p, lmax=lmax_sinv)

print("---- ivfs:cinv")
# Here we do the inversion in iterations, eps_min is the convergence criterion
# Inverse variance filtered lensed sims (and data)
cinv_len = sl.cinv.CinvFilt(obs_len, # Signal and noise
                            sinvfilt, # Signal filter
                            ninvfilt, # Noise filter
                            lib_dir=bdir+"cinv_len_t1p1/", # Location for storing covariance inverse filtered maps
                            eps_min=5e-5,)
# Mask out the high ells
cinv_len = sl.cinv.CinvFiltMasked(lmin=lmin_cinv, lxmin=lxmin_cinv, lmax=lmax_cinv, cinv=cinv_len)

# Inverse variance filtered unlensed sims
cinv_unl = sl.cinv.CinvFilt(obs_unl,
                            sinvfilt,
                            ninvfilt,
                            lib_dir=bdir+"cinv_unl_t1p1/",
                            eps_min=5e-5,)
# Mask out the high ells
cinv_unl = sl.cinv.CinvFiltMasked(lmin=lmin_cinv, lxmin=lxmin_cinv, lmax=lmax_cinv, cinv=cinv_unl)

# Lensed sims with no foregrounds
cinv_len_t2_nofg = sl.cinv.CinvFilt(obs_len_t2_nofg,
                                    sinvfilt,
                                    ninvfilt,
                                    lib_dir=bdir+"cinv_len_t2p1_nofg/",
                                    eps_min=5e-5,)
cinv_len_t2_nofg = sl.cinv.CinvFiltMasked(lmin=lmin_cinv, lxmin=lxmin_cinv, lmax=lmax_cinv, cinv=cinv_len_t2_nofg)

# Lensed sims with no foregrounds
cinv_len_t1_nofg = sl.cinv.CinvFilt(obs_len_t1_nofg,
                                    sinvfilt,
                                    ninvfilt,
                                    lib_dir=bdir+"cinv_len_t1p1_nofg/",
                                    eps_min=5e-5,)
cinv_len_t1_nofg = sl.cinv.CinvFiltMasked(lmin=lmin_cinv, lxmin=lxmin_cinv, lmax=lmax_cinv, cinv=cinv_len_t1_nofg)

# Separate these out, since different numbers of sims are used for different purposes
ivflibs = [cinv_len, cinv_len_t1_nofg, cinv_len_t2_nofg] # Everything, used for get_dat_teb()
ivflibs_mc_sims = [cinv_len] # Evaluate for idxs in mc_sims
ivflibs_mc_sims_n1 = [cinv_len_t2_nofg, cinv_len_t1_nofg] # Evaluate for idxs in mc_sims_mf
ivflibs_mc_sims_unl = [cinv_unl] # Evaluate for idxs in mc_sims_unl

# Libraries we use to calculate quadratic estimates of phi, qest
print("---- qest")
# Data (or sims treated identically to data)
# Define the qest object; the initialization requires the lensed and unlensed cls, as well as the cinv object
qest_dd = sl.quadest.QuadEstLib(cl_unl, cl_len, cinv_len,
                                lib_dir=bdir+'par_%s/qest_len_dd/' % qest_prefix)

# Data x Sim, used for rdn0
qest_ds = sl.quadest.QuadEstLib(cl_unl, cl_len, cinv_len,
    				ivfs2=sl.cinv.CinvFiltData(cinv_len),
    				lib_dir=bdir+'par_%s/qest_len_ds/' % qest_prefix)

# simA x simB, used for N0
qest_ss = sl.quadest.QuadEstLib(cl_unl, cl_len, cinv_len,
    				ivfs2=sl.cinv.CinvFiltSim(cinv_len, roll=2),
    				lib_dir=bdir+'par_%s/qest_len_ss/' % qest_prefix)

# sim(t1p1) x sim(t2p1)
qest_ss2_nofg = sl.quadest.QuadEstLib(cl_unl, cl_len, cinv_len_t1_nofg,
    				      ivfs2=sl.cinv.CinvFiltSim(cinv_len_t2_nofg, roll=0),
    				      lib_dir=bdir+'par_%s/qest_len_ss2_nofg/' % qest_prefix)

# simA x simB, used to subtract from N1
qest_ss_nofg = sl.quadest.QuadEstLib(cl_unl, cl_len, cinv_len_t1_nofg,
    				     ivfs2=sl.cinv.CinvFiltSim(cinv_len_t1_nofg, roll=2),
    				     lib_dir=bdir+'par_%s/qest_len_ss_nofg/' % qest_prefix)

# Unlensed sims (unlensed input spectrum)
qest_uu = sl.quadest.QuadEstLib(cl_unl, cl_len, cinv_unl,
    				lib_dir=bdir+'par_%s/qest_unl_uu/' % qest_prefix)

# Used for N0 for unlensed sims                                    
qest_uu_n0 = sl.quadest.QuadEstLib(cl_unl, cl_len, cinv_unl,
    				   ivfs2=sl.cinv.CinvFiltSim(cinv_unl, roll=2),
    				   lib_dir=bdir+'par_%s/qest_unl_uu_n0/' % qest_prefix)

# Used for MC response
qest_kappa = sl.quadest.QuadEstLibKappa(qest_dd, cmblib_kap)

qftlibs = [qest_dd, qest_ds, qest_ss, qest_uu, qest_ss2_nofg, qest_ss_nofg, qest_kappa] #, qest_ss_nofg, qest_ss2_nofg ]

# qecl
print("---- qecl")
qecl_len_dd = sl.quadest_cl.QuadEstCl(qest_dd,
                                      lib_dir=bdir+'par_%s/qecl_len_dd/' % qest_prefix,
    				      mc_sims_mf=mc_sims_mf)

qecl_len_ds = sl.quadest_cl.QuadEstCl(qest_ds,
                                      lib_dir=bdir+'par_%s/qecl_len_ds/' % qest_prefix,
     				      mc_sims_mf=None)

qecl_len_ss = sl.quadest_cl.QuadEstCl(qest_ss,
                                      lib_dir=bdir+'par_%s/qecl_len_ss/' % qest_prefix,
  				      mc_sims_mf=None)

qecl_len_ss_nofg = sl.quadest_cl.QuadEstCl(qest_ss_nofg,
                                      	   lib_dir=bdir+'par_%s/qecl_len_ss_nofg/' % qest_prefix,
  					   mc_sims_mf=None)

qecl_len_ss2_nofg = sl.quadest_cl.QuadEstCl(qest_ss2_nofg,
                                      	    lib_dir=bdir+'par_%s/qecl_len_ss2_nofg/' % qest_prefix,
 					    mc_sims_mf=None)

qecl_len_uu = sl.quadest_cl.QuadEstCl(qest_uu,
                                      lib_dir=bdir+'par_%s/qecl_len_uu/' % qest_prefix,
   				      mc_sims_mf=mc_sims_mf)

qecl_len_uu_n0 = sl.quadest_cl.QuadEstCl(qest_uu_n0,
                                         lib_dir=bdir+'par_%s/qecl_len_uu_n0/' % qest_prefix,
   					 mc_sims_mf=None)

qecl_len_dk = sl.quadest_cl.QuadEstCl(qest_dd, lib_dir=bdir+'par_%s/qecl_len_dk/' % qest_prefix, qeB=qest_kappa)

qcllibs = [
    qecl_len_dd,
    qecl_len_ds,
    qecl_len_ss,
    qecl_len_uu,
    qecl_len_ss_nofg,
    qecl_len_ss2_nofg,
    qecl_len_uu_n0,
    qecl_len_dk
]
