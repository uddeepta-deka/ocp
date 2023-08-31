#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 10:29:54 2020

@author: uddeepta
"""

import numpy as np
import numpy.random as npr
import random as rd
import matplotlib.pyplot as plt


alpha = 1
beta = 1
a = 1.0 #[-a,a] denotes the interval
mu = 0.
N = 500 #number of particles/eigenvalues
nr = 100000 #number of realizations


def en(x):
    'Calculating the energy for a given set x'
    
    count = 0 #it counts the number of eigenvalues within the interval
    Ex = 0 
    for i in range(N):
        Ex = Ex - 2.*alpha*(2*i-N-1)*x[i] + N * 0.5 * (x[i]**2)
        if -a <= x[i] <= a:
            count+=1.0
    Ex = Ex + (mu * count)
    return Ex


def x_calc():
    xi = npr.normal(0.,1.,N)  #initial configuration
    eni = en(xi)
    st =[]
    x = []
    e = []
    AccRejRatio = []
    accept = 0
    'Monte-Carlo step: find the config after nr number of realizations'    
    for step in range(nr):
        xn = xi.copy()
        loc = npr.randint(0,N) #generates location of the eigenvalue to be changed in next iteration
        xn[loc] = xn[loc] + npr.uniform(-1/np.sqrt(N),1/np.sqrt(N)) #new configuration
        enn = en(xn)
        dE = enn-eni
        
        #Metropolis-Hastings step
        r = rd.random()
        if np.log(r)<min(0,-beta*dE):
            xi = xn
            eni=en(xi)
            accept+= 1
        
        AccRejRatio.append(accept/step)
        
        if step % (2*N)==0:
            st.append(step)
            x.append(xi)
            e.append(eni)
            
    return st,x,e,AccRejRatio

step, config, energy, Acc_Rej = x_calc()

#plt.hist(x_flat, bins = 50, density = True)
#plt.show()
#np.save('mu0_new.npy',x_flat)
