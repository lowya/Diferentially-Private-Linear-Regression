#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:51:23 2021

@author: andrewlowy
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns; sns.set()
import os
import sklearn
import tensorflow as tf
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import scipy
import itertools
from sklearn.utils import shuffle


#####User params to set: 
#j (for choosing which beta parameter to test)
#experiment = 1 to optimize hyperparameters vs. experiment = 2 to plot results for all parameter combos
#bet for true value of beta in indices where beta is nonzero (only matters when H is false)

######Generate Data######
np.random.seed(0)
n = 1000 
#d = 3
d = 10 #9 features, 1 intercept
bet = 0.5 
##Y = X*beta + eta normal
#fix beta to have 3 zero compnents (1,4,7) and the rest are set equal to bet (nonzero)
#if j in 1,4,7: H: beta = 0 is true; otherwise H is false
beta = np.zeros(d)
for i in range(d):
    if i not in [1, 4, 7]:
        beta[i] = bet
    #if i==2:
        #beta[i] = bet
#Generate X randomly 
X = np.random.normal(loc=0.0, scale=1.0, size=(n,d))
#first columns ones:
X[:, 0] = np.ones(n)


#sigma = 1
eta = np.random.normal(loc = 0, scale = 1, size = n)

Y_true = X@beta
Y = Y_true + eta

X = pd.DataFrame(X)
Y = pd.DataFrame(Y, columns=["Y"])

partitions = {} #for each M, store a partition (ie list [D_1, ... D_M)]) of data into M groups 
###To control for randomness in partitioning across runs of experiment with diff eps###
Ms = [1, 2, 10, 20, 40, 50, 75]
for M in Ms:
    partitions[M] = []
    frames = [X, Y]
    df = pd.concat(frames, axis = 1)
    df = shuffle(df)
    J = math.floor(n/M)
    for m in range(M):
        partitions[M].append(df.iloc[m*J: (m+1)*J])

        
####Barrientos et al. (2019) Alg 1: DP Test Stat####
def t(X, Y, n, eps, a, M, j, partition):
    '''
    Parameters
    ----------
    X : feature data
    Y : target data
    n : number of data point x,y pairs
    eps : privacy param > 0
    a : truncation threshold 
    M : number of subgroups of dataset for sample and aggregate 
    j : index for test of significance (in range(d))
    partition: list of len M; partition of Data into M subgroups 

    Returns
    -------
   dp Test stat for test of significacne of beta_j

    '''
    #randomly partition data set into M groups:
    #frames = [X, Y]
    #df = pd.concat(frames, axis = 1)
    #df = shuffle(df)
    #J = math.floor(n/M)
    T = 0
    for m in range(M):
        #S = df.iloc[m: m+J]
    #compute non-private T stat for each subgroup of data
        S = partition[m]
        mod = sm.OLS(S["Y"], S.iloc[:, :10])
        res = mod.fit()
        t = res.tvalues[j]
        if t < -a:
            t = -a
        elif t > a:
            t = a
        T += t
    l = np.random.laplace(0, 2*a/(eps*math.sqrt(M))) #generate lap noise
    T = (T/math.sqrt(M)) + l
    return T

###Barrientos et al. (2019) Alg 2: Monte Carlo p-value simulation####
def pval(T, n, d, eps, a, M, N):
    '''
    Parameters
    ----------
    T : dp T value.
    eps : privacy param.
    a : truncation threshold
    M : number of subgroups
    N : number of MonteCarlo rounds

    Returns
    -------
    p-value = proportion of |simulated vals| > |T|

    '''
    df = math.floor(n/M) - d
    count = 0
    for k in range(N):
        studentTs = np.random.standard_t(df, size=M)
        U = 0
        for m in range(M):
            t = studentTs[m]
            if t < -a:
                t = -a
            elif t > a:
                t = a
            U += t
        l = np.random.laplace(0, 2*a/(eps*math.sqrt(M))) #generate lap noise
        s = (U/math.sqrt(M)) + l
        if abs(s) > abs(T):
            count+=1
    pv = count/N
    return pv
        

#Run experiment: for each eps, optimize for choice of a and M (by minimizing dist btwn Tpriv and Tnonpriv)
#compute |Tpriv - Tnonpriv| and p-values for each 
#Plot results vs. eps 
#j = 1 #for j = 1,4,7, null is true, i.e. beta = 0; for the rest, null is false (ie beta_j is nonzero)
j = 1
N = 100000
mod = sm.OLS(Y, X)
res = mod.fit()
nonpriv_T  = res.tvalues[j]
nonpriv_p = res.pvalues[j]
epsilons = [0.25, 0.5, 1, 2.5, 5, 10, 20]
As = [0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 60, 80, 100]
#Ms = [1, 2, 10, 20, 40, 50, 75]
Ts_temp = {} 
Ts_best = {}
best_params = {}
#pvals_temp = {}
pvals_best = {}
Tdiffs_temp = {}
Tdiffs = {}
nreps = 5 #technically, by basic comp theorem, this makes the full alg nreps*eps-DP
prod = list(itertools.product(As, Ms))


experiment = 1 #1 or 2

Ts = {} #for exp2 store list of Ts of len(epsilons) for each (a, M) pair 
Ps = {}

#Caution: plot titles are not automated. Must change title depending on choice of j
#(i.e. depending on whether or not H: beta_j = 0 is true or false)
if experiment == 1:
#Experiment 1: optimize for a and M as fxn of eps:
    for eps in epsilons:
        for i, (a, M) in enumerate(prod):
        #Ts_temp[(a, M)] = 0
        #for rep in range(nreps):
            #Ts_temp[(a, M)] += t(X, Y, n, eps, a, M, j)/nreps
            Ts_temp[(a, M)] = t(X, Y, n, eps, a, M, j, partitions[M])
            Tdiffs_temp[(a, M)] = abs(abs(Ts_temp[(a, M)]) - abs(nonpriv_T))
        #pvals_temp[(a, M)] = p(Ts_temp[(a, M)], eps, a, M, N)
        best_params[eps] = min(Tdiffs_temp, key=Tdiffs_temp.get)
        Ts_best[eps] = Ts_temp[best_params[eps]]
        pvals_best[eps] = pval(Ts_best[eps], n,d, eps, a, M, N)
        print(best_params[eps], "= opt parameters for eps{:f}".format(eps))
        Tdiffs[eps] = Tdiffs_temp[best_params[eps]]
        if abs(abs(nonpriv_T) - abs(Ts_best[eps])) != Tdiffs_temp[best_params[eps]]:
            print("possible mistake")
        Ts_temp = {}
        Tdiffs_temp = {}
        
    #Plot ||Tnonpriv| - |Tpriv|| vs. eps
    fig = pl.figure()
    ax = fig.add_subplot(111)
#ax.plot(epsilons, [test_RSS]*len(epsilons), label = 'Non-private OLS')
    ax.plot(*zip(*sorted(Tdiffs.items())))
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel(r'$||T| - |T_{\epsilon}||$')
    #ax.set_title(r'T-test for $H: \beta = 0$ false (beta = {:f})'.format(bet))
    ax.set_title(r'T-test for $H: \beta = 0$ true')
    #pl.savefig('Final Project' + 'DP_hyptest_Tdiffs_Htrue.png', dpi=400)
    #pl.savefig('Final Project' + 'DP_hyptest_Tdiffs_Hfalse.png', dpi=400)
    pl.show()

#Plot p-values
    fig2 = pl.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title(r'T-test for $H: \beta = 0$ true') 
    #ax2.set_title(r'T-test for $H: \beta = 0$ false (beta = {:f})'.format(bet)) 
    ax2.plot(epsilons, [nonpriv_p]*len(epsilons), label = 'Non-private T-test')
    ax2.plot(*zip(*sorted(pvals_best.items())), label = r'$\epsilon$-DP T-test')
    handles,labels = ax2.get_legend_handles_labels()
    ax2.set_xlabel(r'$\epsilon$')
    ax2.set_ylabel('p-value')
    ax2.legend(handles, labels, loc='upper right')
    #pl.savefig('Final Project' + 'DP_hyptest_pvals_Htrue.png', dpi=400)
    #pl.savefig('Final Project' + 'DP_hyptest_pvals_Hfalse.png', dpi=400)
    pl.show()


else:
#Experiment 2: do experiment with same a, M for each eps instead of optimizing
    for i, (a, M) in enumerate(prod):
        Ts[(a, M)] = []
        Ps[(a, M)] = []
        for e, eps in enumerate(epsilons):
            Ts[(a, M)].append(t(X, Y, n, eps, a, M, j, partitions[M]))
            Ps[(a, M)].append(pval(Ts[(a, M)][e], n, d, eps, a, M, N))
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(epsilons, abs(abs(np.array(Ts[(a,M)])) - abs(np.array([nonpriv_T]*len(epsilons)))))
        ax.set_xlabel(r'$\epsilon$')
        ax.set_ylabel(r'$||T| - |T_{\epsilon}||$')
        ax.set_title(r'T-test for $H: \beta = 0$ true, a = {:f}, M = {:d}'.format(a, M))
        #ax.set_title(r'T-test for $H: \beta = 0$ false ($\beta =$ {:f}), a = {:f}, M = {:d}'.format(bet, a, M))
        #pl.savefig('Final Project' + 'DP_hyptest_Tdiffs_Htrue_exp2.png', dpi=400)
        #pl.savefig('Final Project' + 'DP_hyptest_Tdiffs_Hfalse_exp2.png', dpi=400)
        pl.show()

#Plot p-values
        fig2 = pl.figure()
        ax2 = fig2.add_subplot(111)
        ax2.set_title(r'T-test for $H: \beta = 0$ true, a = {:f}, M = {:d}'.format(a, M))
        #ax2.set_title(r'T-test for $H: \beta = 0$ false ($\beta =${:f}), a = {:f}, M = {:d}'.format(bet, a, M))
        ax2.plot(epsilons, [nonpriv_p]*len(epsilons), label = 'Non-private T-test')
        ax2.plot(epsilons, Ps[(a, M)], label = r'$\epsilon$-DP T-test')
        handles,labels = ax2.get_legend_handles_labels()
        ax2.set_xlabel(r'$\epsilon$')
        ax2.set_ylabel('p-value')
        ax2.legend(handles, labels, loc='upper right')
        #pl.savefig('Final Project' + 'DP_hyptest_pvals_Htrue_exp2.png', dpi=400)
        #pl.savefig('Final Project' + 'DP_hyptest_pvals_Hfalse_exp2.png', dpi=400)
        pl.show()   


            
    
    
    
    