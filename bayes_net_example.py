#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:49:21 2018

@author: miller
"""

import numpy as np
import pymc3 as pm
import pandas as pd

##### Bayes Net Example ######

### Two problems: 
    
#1. Probability of Disease A given Disease B = False, Symptoms = True

#2. Probability of Exposure given Disease A = True, Symptoms = True


niter = 100000  # Number of iterations
tune = 5000  # Burn-In iterations

model = pm.Model() # Initalize model

with model:
    tv = [1] # Initial value
    
    exposure = pm.Bernoulli('exposure', 0.2, shape=1) # Exposure is a bernoulli RV with p = 0.2
    
    risk = pm.Bernoulli('risk', 0.3, shape=1) # Risk is a bernoulli RV with p = 0.3
    
    ### If risk is true and exposure is true, probability of disease a = 0.7
    ### If risk is true and exposure is false, probabiliy of disease a = 0.6
    ### If risk is false and exposure is true, probability of disease a = 0.5
    ### IF risk is false and exposure is false, probability of of disease a = 0.1
    dis_a_p = pm.Deterministic('dis_a_p', 
                pm.math.switch(risk, pm.math.switch(exposure, 0.7, 0.6), 
                pm.math.switch(exposure, 0.5, 0.1)))
    dis_a = pm.Bernoulli('dis_a', dis_a_p, shape=1) # Bernoulli RV with p based on dis_a_p
    
    # Disease b has probability of 0.7 if exposure is true, 0.2 if false
    dis_b_p = pm.Deterministic('dis_b_p', pm.math.switch(exposure, 0.7, 0.2))
    dis_b = pm.Bernoulli('dis_b', dis_b_p, shape=1)
    
    ### If dis_a is true and dis_b is true, probability of symptoms = 0.9
    ### If dis_a is true and dis_b is false, probabiliy of symptoms a = 0.6
    ### If dis_a is false and dis_b is true, probability of disease a = 0.4
    ### IF dis_a is false and dis_b is false, probability of of disease a = 0.05
    sym_p = pm.Deterministic('sym_p', 
                pm.math.switch(dis_a, pm.math.switch(dis_b, 0.9, 0.6), 
                pm.math.switch(dis_b, 0.4, 0.05)))
    sym = pm.Bernoulli('sym', sym_p, shape=1)
    
    ### If dis_a is true and dis_b is true, probability of test a = 0.97
    ### If dis_a is true and dis_b is false, probabiliy of test a = 0.85
    ### If dis_a is false and dis_b is true, probability of disease a = 0.2
    ### IF dis_a is false and dis_b is false, probability of of disease a = 0.08    
    test_a_p = pm.Deterministic('test_a_p', 
                pm.math.switch(dis_a, pm.math.switch(dis_b, 0.97, 0.85), 
                                      pm.math.switch(dis_b, 0.2, 0.08)))
    test_a = pm.Bernoulli('test_a', test_a_p, shape=1)
    
    # Starts MCMC
    trace = pm.sample(niter, step=[pm.BinaryGibbsMetropolis([exposure,risk,dis_b,dis_a,sym,test_a])], tune = tune, random_seed=123)

pm.summary(trace) # Prints MCMC statistics

# Extract info from trace data structure into dictionary
results_dict = {
              'Exposure': [1 if ii[0] else 0 for ii in trace['exposure'].tolist() ],
              'Risk Factors': [1 if ii[0] else 0 for ii in trace['risk'].tolist() ],
              'Disease A Prob': [ii[0] for ii in trace['dis_a_p'].tolist()],
              'Disease A': [1 if ii[0] else 0 for ii in trace['dis_a'].tolist()],
              'Disease B Prob': [ii[0] for ii in trace['dis_b_p'].tolist()],
              'Disease B': [1 if ii[0] else 0 for ii in trace['dis_b'].tolist()],
              'Sym Prob': [ii[0] for ii in trace['sym_p'].tolist()],
              'Sym': [1 if ii[0] else 0 for ii in trace['sym'].tolist()],
              'Test A Prob': [ii[0] for ii in trace['test_a_p'].tolist()],
              'Test A': [1 if ii[0] else 0 for ii in trace['test_a'].tolist()]
              }

df = pd.DataFrame(results_dict)

# Boolean mask indicating for which observations Disease B was not present and Symptoms were
bool_array = np.where( (np.array(df['Disease B']==0) & np.array(df['Sym']==1)), True, False)

# Subset df based on bool_array
num_dis_a = np.sum(df.loc[bool_array,'Disease A'])
total = len(df.loc[bool_array,'Disease A'])
p_dis_a_given_cond = num_dis_a/total
print("Probability of disease A | disease B = False, Symptoms = True: " + str(p_dis_a_given_cond))

# Boolean mask indicating for which observations Test A and Symptoms were True 
bool_array = np.where( (np.array(df['Test A']==1) & np.array(df['Sym']==1)), True, False)
num_exposure = np.sum(df.loc[bool_array,'Exposure'])
total = len(df.loc[bool_array,'Exposure']) 
p_exposure_given_cond = num_exposure/total
print("Probability of exposure | disease A = True, Symptoms = True: " + str(p_exposure_given_cond))
