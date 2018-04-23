
# coding: utf-8

# ## One-Way ANOVA (Compare with anovacoagulation.odc from Feb21 folder)

# The implementation here is not exactly anovacoagulation.odc. Instead of applying STZ or CR corner constraints and setting 
# the value of one alpha to be determinate. Here we have set the value of mu0 to be 0

# In[19]:


import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import itertools


# In[2]:


# Data from the odc file
ntotal=24
a=4
times = [62,60,63,59,63,67,71,64,65,66,68,66,71,67,68,68,56,62,60,61,63,64,63,59]
diets = list(map(int,list('1111')+list('222222')+list('333333')+list('44444444')))


# In[3]:


# Making it a dataframe
# This is probably the more used form to store data
coag_data = pd.DataFrame({'times':times,'diets':diets})


# In[61]:


# Model
with pm.Model() as model:
#    mu0 = pm.Normal('mu0',0,1000)
#    alpha0 = pm.Constant('alpha0',0)
    mu = pm.Normal('mu',0,1000,shape=len(coag_data.diets.unique()))

    tau = pm.Gamma('tau',alpha=0.001, beta=0.001)
    #pm.HalfNormal('tau',0,1000)

    sigma = pm.Deterministic('sigma',pow(1/tau,0.5))
    adiff12 = pm.Deterministic('adiff12', mu[0]-mu[1])
    adiff13 = pm.Deterministic('adiff13', mu[0]-mu[2])
    adiff14 = pm.Deterministic('adiff14', mu[0]-mu[3])
    adiff23 = pm.Deterministic('adiff23', mu[1]-mu[2])
    adiff24 = pm.Deterministic('adiff24', mu[1]-mu[3])
    adiff34 = pm.Deterministic('adiff34', mu[2]-mu[3])
    
    # not using mu0+alpha[coag_data.diets-1] because the STZ or CR constraints could not be enforced
    likelihood = pm.Normal('y',mu[coag_data.diets-1],sigma,observed=coag_data.times) 
    trace = pm.sample(3000, chains=1, tune=300)
    
#pm.traceplot(trace)
#plt.show()
pm.summary(trace)


# The value of sigma match with the result from anovacoagulation.odc. The values of mu_0, mu_1, mu_2, mu_3 should match with the values we get for the MAP estimates for mu0 + alpha_1,
# mu0 + alpha_2, mu0 + alpha_3, mu0 + alpha_4 respectively from the anovacoagulation.odc.
