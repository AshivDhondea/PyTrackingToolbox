# -*- coding: utf-8 -*-
"""
Created on 13 October 2017

Library for dynamics functions for deterministic and stochastic linear dynamics.

@author: Ashiv Dhondea
"""
import numpy as np
import math
import MathsFunctions as MathsFn
# ------------------------------------------------------------------------- #
def fn_Generate_STM_polynom(zeta,nStates):
    """
     fn_Generate_STM_polynom creates the state transition matrix for polynomial models 
     of degree (nStates-1) over a span of transition of zeta [s].
     Polynomial models are a subset of the class of constant-coefficient linear DEs.
     Refer to: Tracking Filter Engineering, Norman Morrison.
    """
    stm = np.eye(nStates,dtype=np.float64);
    for yindex in range (0,nStates):
        for xindex in range (yindex,nStates): # STM is upper triangular
            stm[yindex,xindex] = np.power(zeta,xindex-yindex)/float(math.factorial(xindex-yindex));
    return stm;     

def fn_Generate_STM_polynom_3D(zeta,nStates,dimensionality):
    """
    fn_Generate_STM_polynom_3D generates the full state transition matrix for 
    the required dimensionality.
    
    if dimensionality == 3, the problem is 3-dimensional.
    The state vector is thus defined as 
    [x,xdot,xddot,y,ydot,yddot,z,zdot,zddot]
    """
    stm = fn_Generate_STM_polynom(zeta,nStates);
    stm3 = MathsFn.fn_Create_Concatenated_Block_Diag_Matrix(stm,dimensionality-1);
    return stm3;
