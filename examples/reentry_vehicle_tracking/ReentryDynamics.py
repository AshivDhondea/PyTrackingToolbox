"""
Reentry dynamics

This file contains the dynamics functions for the vehicle reentry tracking problem.

Author: Ashiv Dhondea, RRSG, UCT
Date: 01 September 2016
"""
## Import relevant libraries
import numpy as np

## Define the constants from ref [1]
b0 = -0.59783;
H0 = 13.406;
Gm0 = 3.9860e5;
R0 = 6374;

def fnReentry(t,x):
    r = np.linalg.norm(x[0:2]);
    v = np.linalg.norm(x[2:4]);
    b = b0*np.exp(x[4]);
    D = b*np.exp((R0 - r)/H0)*v;
    G = -Gm0/r**3;
    
    xdot = np.zeros_like(x,dtype=np.float64);
    
    xdot[0] = x[2];
    xdot[1] = x[3];
    xdot[2] = D*x[2] + G*x[0];
    xdot[3] = D*x[3] + G*x[1];
    # The SDE method will add the w1,2,3 terms xdot[2,3,4] by multiplying with matrix L.
    return xdot

def fnRadar(x,xradar):
    ymeas = np.zeros([2],dtype=np.float64);
    displacement_x = x[0] - xradar[0];
    displacement_y = x[1] - xradar[1];
    ymeas[0] = np.sqrt(displacement_x**2 + displacement_y**2);
    ymeas[1] = np.arctan2(displacement_y,displacement_x);
    return ymeas
    
def fnRadarObsv(ymeas,R):
    ymeas = ymeas + np.random.multivariate_normal([0.0,0.0],R);
    return ymeas
    
