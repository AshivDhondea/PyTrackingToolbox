# -*- coding: utf-8 -*-
"""
Created on Wed Feb 07 11:59:54 2018

Reentry vehicle tracking

@author: Ashiv Dhondea
"""
"""
First main file. Based on Pj00_reentry.py of 2016.

Based on:
1. @article{julier2004unscented,
  title={Unscented filtering and nonlinear estimation},
  author={Julier, Simon J and Uhlmann, Jeffrey K},
  journal={Proceedings of the IEEE},
  volume={92},
  number={3},
  pages={401--422},
  year={2004},
  publisher={IEEE}
}

2. @article{sarkka2007unscented,
  title={On unscented Kalman filtering for state estimation of continuous-time nonlinear systems},
  author={Sarkka, Simo},
  journal={IEEE Transactions on automatic control},
  volume={52},
  number={9},
  pages={1631--1641},
  year={2007},
  publisher={IEEE}
}

Copyright 2017, 2018 AshivD <ashivdhondea5@gmail.com>

"""
# --------------------------------------------------------------------------- #
## Include necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
params = {'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']}
plt.rcParams.update(params) # updated: 12 April 2017

## Ashiv's own stuff
import Num_Integ as Ni
import ReentryDynamics as Rd
# --------------------------------------------------------------------------- #
## Define time variable
dt = 0.1; # in [s]
t_end = 200;

T = np.arange(0,t_end,dt,dtype=np.float64);

## Define the constants from ref [1]
b0 = -0.59783;
H0 = 13.406;
Gm0 = 3.9860e5;
R0 = 6374;

# Process noise covariance matrix, Qk is the discrete version of Qc, the 
# continuous one
Qk = np.diag([2.4064e-5,2.4064e-5,1e-6]); 
Qc = Qk/dt;
# L : dispersion matrix
L = np.array([[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64);
# Covariance matrix of state vector estimate
true_P0 = np.diag([1e-6,1e-6,1,1e-6,1e-6,0.0]);
true_Qc = Qc;
true_Qc[2,2] = 1e-6;
true_m0 = np.array([6500.4,349.14,-1.8093,-6.7967,0.6932],dtype=np.float64);

m0 = np.array([6500.4,349.14,-1.8093,-6.7967,0.0],dtype=np.float64);
P0 = true_P0; P0[4] = 1.0;

xradar = np.array([R0,0],dtype=np.float64);
## Define state vector
# x[0],x[1] -> x & y position
# x[2],x[3] -> x & y velocity
# x[4] -> parameter of the vehicle's aerodynamic properties
    
x_state = Ni.fnEuler_Maruyama(true_m0,Rd.fnReentry,T,L,true_Qc*dt);
print 'Truth data generated.'
# --------------------------------------------------------------------------- #
fig = plt.figure(1)
ax = fig.gca();
plt.rc('text', usetex=True); plt.rc('font', family='serif');
fig.suptitle(r"\textbf{Reentry tracking problem}")
plt.plot(x_state[0,:],x_state[1,:],label = r'Target trajectory');
aa = 0.02*np.arange(-1,4,0.1,dtype=np.float64);
cx = R0*np.cos(aa);
cy = R0*np.sin(aa);
plt.plot(xradar[0],xradar[1],'k',marker='o',label=r'Radar');
plt.plot(cx,cy,'g',label=r'Earth');
plt.legend(loc='best');
plt.axis([6340,6520,-200,600])
plt.ylabel(r" $y~[\mathrm{km}]$")
plt.xlabel(r" $x~[\mathrm{km}]$")
plt.grid(True,which='both',linestyle=(0,[0.7,0.7]),lw=0.4,color='black')
plt.show()
fig.savefig('main_00_reentry_00_create_scenario.pdf',bbox_inches='tight',pad_inches=0.05,dpi=10)

