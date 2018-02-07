# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 12:22:55 2018

Reentry vehicle tracking

@author: Ashiv Dhondea
"""
# --------------------------------------------------------------------------##
## Include necessary libraries
import numpy as np
# Importing what's needed for nice plots.
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Helvetica']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params)

## Ashiv's own stuff
import Num_Integ as Ni
import ReentryDynamics as Rd
# -------------------------------------------------------------------------#
## Define time variable
dt = 0.1; # in [s]
t_end = 200;
T = np.arange(0,t_end,dt,dtype=np.float64);
## Define the constants from ref [1]
b0 = -0.59783;
H0 = 13.406;
Gm0 = 3.9860e5;
R0 = 6374.0;

Qk = np.diag([2.4064e-5,2.4064e-5,1e-6]); 
Qc = Qk/dt;

# L : dispersion matrix
L = np.array([[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64);

true_P0 = np.diag([1e-6,1e-6,1e-6,1e-6,0.0]);
true_Qc = Qc;
true_Qc[2,2] = 1e-6;
true_m0 = np.array([6500.4,349.14,-1.8093,-6.7967,0.6932],dtype=np.float64);

m0 = np.array([6500.4,349.14,-1.8093,-6.7967,0.0],dtype=np.float64);
P0 = true_P0; P0[4] = 1.0;

# Location of radar
xradar = np.array([R0,0],dtype=np.float64);
sd_range = 1.0; # [km]
sd_bearing = 0.17e-3; # [rad]
R = np.diag([sd_range**2,sd_bearing**2]); # Radar covariance matrix from Julier 2004
## ------------------------------------------------------------------------##
## Define state vector
# x[0],x[1] -> x & y position
# x[2],x[3] -> x & y velocity
# x[4] -> parameter of the vehicle's aerodynamic properties

# Generate truth data with an Euler-Mauryama scheme (order 0.5 strong)    
x_state = Ni.fnEuler_Maruyama(true_m0,Rd.fnReentry,T,L,true_Qc*dt);
print 'Truth data generated.'

dy = 2; # the radar measures 2 quantities
y = np.zeros([dy,len(T)],dtype=np.float64);
ymeas = np.zeros([dy,len(T)],dtype=np.float64);

for index in range(0,len(T)):
    y[:,index] = Rd.fnRadar(x_state[:,index],xradar);
    ymeas[:,index] =  Rd.fnRadarObsv(y[:,index],R);
# ------------------------------------------------------------------------- #
f, axarr = plt.subplots(2,sharex=True);
plt.rc('text', usetex=True)
plt.rc('font', family='serif');
plt.rc('font',family='helvetica');
f.suptitle(r"\textbf{Reentry vehicle tracking: radar measurements}" ,fontsize=12)
axarr[0].plot(T,y[0,:],label=r'no process noise')
axarr[0].plot(T,ymeas[0,:],label=r'with measurement noise')
axarr[0].set_ylabel(r'Range $\rho~[\mathrm{km}]$');
axarr[1].plot(T,y[1,:],label=r'no process noise')
axarr[1].plot(T,ymeas[1,:],label=r'with measurement noise')
axarr[1].set_ylabel(r'Bearing $\theta~[\mathrm{rad}]$');
axarr[1].set_xlabel(r'Time $t~[\mathrm{s}]$');
axarr[0].grid(True)
axarr[1].grid(True)
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0:1]], visible=False)
axarr[0].legend(loc="upper left",shadow=True, fancybox=True)
axarr[1].legend(loc="upper left",shadow=True, fancybox=True)
f.savefig('main_00_reentry_01_noise.pdf',bbox_inches='tight',pad_inches=0.11,dpi=10)
# ------------------------------------------------------------------------- #