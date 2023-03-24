#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import sys
from scipy.optimize import curve_fit
from random import seed
from random import random
from random import randint
import random
import time

#n = int(input("No of coin tosses per trial [e.g.10]: "))
n = 10 # set to 10 tosses
t = int(input("No of trials [e.g. 10]: "))

q = 0
results = []
for q in range(t):  

    #print("===== TRIAL %d =======" %(q))
    random.seed(time.time()) # 'randomise' the seed each time

    '''
    data = [random.randint(0, 1) for _ in range(n)]
    print(data) # produces a single array
    '''

    i =0
    for j in range(n):  
        value = randint(0, 1) # generate random intergers - 0 and 1
        #print(value) # WANT TO COUNT NUMBER OF HEADS, THEN META THE TRIALS
        if value == 1:
            i=i+1
        #print('Toss %d gives %d' %(j,value))

    #print('Trial %d - %d tosses gives %d heads' %(q,n,i)) # need to save these
    results.append(i)

mean = np.mean(results)
std = np.std(results)
med = np.median(results)
N = len(results)
########### TRY FITTING A GAUSSAN ################
from scipy.stats import norm
import scipy.stats as stats

df = pd.DataFrame(results, columns=['heads'])
print(df.tail())
counts= df.groupby(['heads']).size()
#print(counts) # series or dataframe?
df2 = counts.to_frame().reset_index()
#print(df2)
x = df2['heads']
y= df2.iloc[:, 1] # 2nd column as this no title thing a pain
#print(x,y)

norm_est = t/4
m_est = mean
s_est = std # inital guess

def fit_func(x,norm,m,s): # x DATA THEN GUESSES
   return norm*np.exp(-1*(x-m)**2/(2*s*s))

p0 = [norm_est,m_est,s_est] 
parameters, covariance = curve_fit(fit_func, x,y,p0)
norm = parameters[0]
m = parameters[1]
s = parameters[2]

SE = np.sqrt(np.diag(covariance))
dm = SE[1]
ds = SE[2]

print('Fit gives norm = %1.1f mean = %1.2f +/- %1.2f , sigma = %1.2f  +/- %1.2f ' % (norm, m,dm,s, ds)) 

xplot = np.linspace(0, 10,100) 
fit_y = fit_func(xplot,norm,m,s)
##################################################
print('%d x %d tosses => mean = %1.2f, sigma = %1.2f median = %1.1f' % (t,n,mean,std,med))

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(6,4))
ax = plt.gca()
plt.setp(ax.spines.values(),linewidth=2)
  
text = 'No. of heads per %d tosses' %(n)
plt.ylabel('Number', size=14); plt.xlabel(text, size=14)
ax.hist(results, bins=10, color="w", edgecolor='r', linewidth=3, range = [0.5,10.5]);
ax.plot(xplot, fit_y, '-', c = 'b',linewidth=3) 
# EXTEND Y-AXIS SLIGHTLY##
ymin, ymax = plt. ylim(); #print(ymax) 
text = "\u03BC = %1.2f, \u03C3 = %1.2f, %d x %d tosses" %(mean,std,t, n)
plt.text(0,1.25*ymax, text, fontsize = 12, c = 'r', horizontalalignment='left',verticalalignment='top') 
text = "\u03BC = %1.2f $\pm$ %1.2f, \u03C3 = %1.2f $\pm$ %1.2f" %(m,dm,s,ds)
plt.text(0,1.15*ymax, text, fontsize = 12, c = 'b', horizontalalignment='left',verticalalignment='top') 
plt.ylim(ymin * 1, ymax * 1.3)
plt.tight_layout()
# png = "random-horm_histo-t=%d_n=%d.png" % (t,n) 
# plt.savefig(png);print("Plot written to", png)
plt.show()

