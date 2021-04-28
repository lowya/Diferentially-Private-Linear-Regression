#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 13:04:20 2021

@author: andrewlowy
"""

##CONTROL FOR RANDOMNESS IN SAMPLING BY SAMPLING 1000 TIMES ONCE AT THE BEGINNING##

######## DP Lin Reg ############
#Obj pert vs. noisy SGD with amp by sampling vs. non-private OLS
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:46:38 2021

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


np.random.seed(0)


#EXPLORATORY DATA ANALYSIS 
#Open data as dataframe in pandas 
df = pd.read_csv('insurance.csv')
df.describe()


#encode categorical features
from sklearn.preprocessing import LabelEncoder
#sex
le = LabelEncoder()
le.fit(df.sex.drop_duplicates()) 
df.sex = le.transform(df.sex)
# smoker or not
le.fit(df.smoker.drop_duplicates()) 
df.smoker = le.transform(df.smoker)
#region
le.fit(df.region.drop_duplicates()) 
df.region = le.transform(df.region)

#inspect correlations:
df.corr()['charges'].sort_values()
f, ax = pl.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(240,10,as_cmap=True),
            square=True, ax=ax)

#seperate features and target 
X = df.iloc[::, 0:6]
X.insert(0, 'const', 1)
Y = df.iloc[::, 6]



#SEPERATE TRAIN AND TEST DATA SETS SO WE CAN MEASURE OUT OF SAMPLE ACCURACY
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

#STANDARDIZE (NON-CATEGORICAL) FEATURES-age, bmi:
standardized_Xtrain_cols = (X_train.iloc[::, 1:4:2] - X_train.iloc[::, 1:4:2].mean())/X_train.iloc[::, 1:4:2].std()
standardized_Xtest_cols = (X_test.iloc[::, 1:4:2] - X_train.iloc[::, 1:4:2].mean())/X_train.iloc[::, 1:4:2].std()

X_test = X_test.assign(age =standardized_Xtest_cols['age'])
X_test = X_test.assign(bmi =standardized_Xtest_cols['bmi'])
#y_test = (y_test - y_train.mean())/y_train.std()

X_train = X_train.assign(age =standardized_Xtrain_cols['age'])
X_train = X_train.assign(bmi =standardized_Xtrain_cols['bmi'])
#y_train = (y_train - y_train.mean())/y_train.std()


#NON-PRIVATE BASELINES:
mod = sm.OLS(y_train, X_train)
res = mod.fit()
print(res.summary()) 
#R^2 = 0.742 = percent of variation in Y explained by linear model 
#tiny p-val for F-stat for test of significance  -> suggests lin model is reasonable for this data
#tiny p-values for all coefficients except sex and region

rss = res.ssr
print('train RSS is', rss) 

#S^2 = rss/(n-p) #df = 1063 = n - p (for training data)
Ssquared = rss/1063
print('train Ssquared is', Ssquared)
#8.04329181985294

pl.clf()
residuals = res.resid
pl.scatter(residuals.index, residuals.values)
pl.ylabel("Residuals")
pl.xlabel("Data index")
pl.title("OLS Residuals for Insurance Data (train)")


#make predictions on test data: 
yhat_test =  res.predict(X_test)
test_residuals = y_test - yhat_test 

#test error (test RSS):
test_RSS = np.linalg.norm(test_residuals)**2
print('test_RSS is', test_RSS) 
test_Ssquared = test_RSS/(268-7)
print('test_Ssquared is',test_Ssquared) #interestingly, it's a bit smaller than training Ssquared 

#plot test errors:
pl.scatter(test_residuals.index, test_residuals.values)
pl.ylabel("Residuals")
pl.xlabel("Data index")
pl.title("OLS Residuals for Insurance Data (test: orange; train: blue)")


N = X_train.shape[0] 
d = X_train.shape[1]
#DP SGD with gradient clipping (since lip constant is huge otherwise due to Y):
#Preliminary functions:
def squared_loss(w, x, y):
    return (y - np.dot(w,x))**2 

def squared_loss_gradient(w, x, y):
    return -2*x*(y - np.dot(w,x))

def gauss_mech2(d, N, eps, delta, L): #L is  clip threshold #non-local - use Adv Comp and privacy amp by sampling 
    return np.random.multivariate_normal(mean = np.zeros(d), cov = (8*T*(L**2)*(np.log(1/delta) * np.log(T/delta))/((eps*N)**2))*np.eye(d))

def F_eval(w, X, Y): #evaluate full empirical loss (put in w_hat to get RSS)
    return np.linalg.norm(Y - X@w)**2


#Make list S of 1000 (not necessarily distinct) indices, 1000 uniform random draws from indices
S = [0]*1000
for t in range(1000):
    S[t] = np.random.randint(0,X_train.shape[0])

def noisy_sgd(eps, delta, L, d, N, T, stepsize, loss_freq): #inspired by Bassily et al. 2014
    avg_window = loss_freq #user choice 
    losses = []
    iterates = [np.zeros(d)] #list storing d-dim arrays (iterates), initialized with 0 vector
    g = np.zeros(d) #initialize with 0 
    for t in range(T):
        i = S[t] #sample random datapoint  
        x, y = X_train.iloc[i], y_train.iloc[i]
        c = min(1, L/np.linalg.norm(squared_loss_gradient(iterates[-1], x, y))) #gradient clippping 
        b = squared_loss_gradient(iterates[-1], x, y)*c
        g = b + gauss_mech2(d, N, eps, delta, L) #g is noisy clipped MB grad of loss at last iterate 
        iterates.append(iterates[-1] - stepsize * g) #take SGD step and add new iterate to list iterates 
    loss_vals = [0]*len(iterates)
    for j, iterate in enumerate(iterates):
        loss_vals[j] = F_eval(iterate, X_train, y_train)
    best_index = np.argmin(loss_vals)
    loss = min(loss_vals)
    best_iterate = iterates[best_index]
    return best_iterate, loss, 'converged' #returns best iterate

def sgd(L, d, N, T, stepsize, loss_freq): 
    iterates = [np.zeros(d)] #list storing d-dim arrays (iterates), initialized with 0 vector
    g = np.zeros(d) #initialize with 0 
    for t in range(T):
        i = S[t] #sample random datapoint  
        x, y = X_train.iloc[i], y_train.iloc[i]
        c = min(1, L/np.linalg.norm(squared_loss_gradient(iterates[-1], x, y))) #gradient clippping 
        b = squared_loss_gradient(iterates[-1], x, y)*c
        g = b #g is clipped grad of loss at last iterate 
        iterates.append(iterates[-1] - stepsize * g) #take SGD step and add new iterate to list iterates 
    loss_vals = [0]*len(iterates)
    for j, iterate in enumerate(iterates):
        loss_vals[j] = F_eval(iterate, X_train, y_train)
    best_index = np.argmin(loss_vals)
    loss = min(loss_vals)
    best_iterate = iterates[best_index]
    return best_iterate, loss, 'converged' #returns best iterate


Ls = [10, 100, 1000, 10000, 100000, 1000000, 1000000, 99999999999] #clip thresholds
T = 1000  #user can modify
epsilons = [0.5, 1, 2.5, 5, 10, 20]
delta = 1/X_train.shape[0]
loss_freq = 10
n_reps = 3
n_stepsizes = 10
stepsizes = [np.exp(exponent) for exponent in np.linspace(-7,0,n_stepsizes)]
stepLproduct = list(itertools.product(stepsizes, Ls))
sgd_loss_temp = {}
nonpriv_loss_temp = {}
sgd_w_temp = {}
nonpriv_w_temp = {}
sgd_w = {} #stores optimal w for each epsilon
nonpriv_w = {}
#eps_loss = {}
#test_RSS_objpert = {} #will store corresponding RSS for each w(eps)
test_RSS_sgd = {} #store final test RSS (for optimal stepsize) each eps 
train_RSS_sgd = {}
sgd_predictions = {}
sgd_residuals_train = {}
nonprivSGD_predictions = {}
nonprivSGD_residuals_train = {}
nonprivSGD_train_RSS = {}
nonprivSGD_test_RSS = {}
nonprivSGD_residuals_train = {}
#Goal: get plot of test_RSS vs. epsilon & include non-private on plot)


for eps in epsilons:
    path = 'dp_linreg_eps{:f}'.format(eps)
#tune stepsize and clip threshold for Noisy SGD:
    for i, (step, L) in enumerate(stepLproduct):
        sgd_loss_temp[(step, L)] = 0
        sgd_w_temp[(step, L)] = np.zeros(d)
        nonpriv_loss_temp[(step, L)] = 0
        nonpriv_w_temp[(step, L)] = np.zeros(d)
        for rep in range(n_reps): #n_reps=3 trials for each stepsize
            w, l, success = noisy_sgd(eps, delta, L, d, N, T, step, loss_freq)
            sgd_loss_temp[(step, L)] += l / n_reps #each rep we add average risk vals (for all T//loss_freq iterates) over the n_reps= 3 trials to ith col of loss_temp
            sgd_w_temp[(step, L)] += w/n_reps 
            W, ell, succ = sgd(L, d, N, T, step, loss_freq)
            nonpriv_loss_temp[(step, L)] += ell/n_reps 
            nonpriv_w_temp[(step, L)] += W/n_reps
            #after all n_reps reps, sgd_loss_temp[(step,L)] = avg loss for (step,L) & sgd_w[i] = best iterate w for (step,L)         
    m = min(sgd_loss_temp, key=sgd_loss_temp.get) #optimal (stepsize, L) that minimizes loss for eps
    M = min(nonpriv_loss_temp, key=nonpriv_loss_temp.get)
    print(m, "= opt parameters for eps{:f}".format(eps))
    print(M, "=opt params for non-private")
    sgd_w[eps] = sgd_w_temp[m] #store best w among all (L,stepsize) (and all T iterates) for each eps
    print("best iterate for eps{:f} is".format(eps), sgd_w[eps])
    nonpriv_w[eps] = nonpriv_w_temp[M]
    
#Use optimal w to make predictions on X_test to obtain sgd_yhat_test and RSS; store RSS in test_RSS_sgd[eps]
    sgd_predictions[eps] = X_test@sgd_w[eps]
    nonprivSGD_predictions[eps] = X_test@nonpriv_w[eps]
    train_RSS_sgd[eps] = F_eval(sgd_w[eps], X_train, y_train)
    nonprivSGD_train_RSS[eps] = F_eval(nonpriv_w[eps], X_train, y_train)
    sgd_residuals_train[eps] = X_train@sgd_w[eps] - y_train
    nonprivSGD_residuals_train[eps] = X_train@nonpriv_w[eps] - y_train
    print(train_RSS_sgd[eps], "= train RSS for eps{:f}".format(eps))
    test_RSS_sgd[eps] = F_eval(sgd_w[eps], X_test, y_test)
    nonprivSGD_test_RSS[eps] = F_eval(nonpriv_w[eps], X_test, y_test)
    print(test_RSS_sgd[eps], "= test RSS for eps{:f}".format(eps))
    #reset temp dicts before moving to next eps
    sgd_w_temp = {} 
    sgd_loss_temp = {}
    nonpriv_w_temp = {}
    nonpriv_loss_temp = {}
#END FOR


#PLOT TEST RSS VS. EPS FOR EACH ALG
fig = pl.figure()
ax = fig.add_subplot(111)
ax.plot(epsilons, [test_RSS]*len(epsilons), label = 'Non-private OLS')
ax.plot(*zip(*sorted(test_RSS_sgd.items())), label='Noisy SGD after {:d} Rounds'.format(T))
ax.plot(*zip(*sorted(nonprivSGD_test_RSS.items())), label='SGD after {:d} Rounds'.format(T))

handles,labels = ax.get_legend_handles_labels()
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('Test RSS')
ax.set_title('DP Linear Regression') 
ax.legend(handles, labels, loc='upper right')
pl.savefig('Final Project' + path + 'DP_test_RSS_vs_epsilon3.png', dpi=400)
pl.show()

#Plots for TRAIN RSS:
fig2 = pl.figure()
ax = fig2.add_subplot(111)
ax.plot(epsilons, [rss]*len(epsilons), label = 'Non-private OLS')
ax.plot(*zip(*sorted(train_RSS_sgd.items())), label='Noisy SGD after {:d} Rounds'.format(T))
ax.plot(*zip(*sorted(nonprivSGD_train_RSS.items())), label='SGD after {:d} Rounds'.format(T))

handles,labels = ax.get_legend_handles_labels()
ax.set_xlabel(r'$\epsilon$')
ax.set_ylabel('Train RSS')
ax.set_title('DP Linear Regression') 
ax.legend(handles, labels, loc='upper right')
pl.savefig('Final Project' + path + 'DP_train_RSS_vs_epsilon3.png', dpi=400)
pl.show()

#Residual plots for SGD and noisy SGD (for a few fixed epsilon values):
nonpriv_res_pl = pl.hist(list(nonprivSGD_residuals_train[1]), bins = 'auto')
pl.title("SGD (Train) Residuals for Insurance Data (T = 1000)")
pl.show()
for eps in epsilons:
    nsgd_res_pl = pl.hist(list(sgd_residuals_train[eps]), bins = 'auto')
    pl.title("DP SGD (Train) Residuals for Insurance Data ($\epsilon$= {:f})".format(eps))
    pl.show()

