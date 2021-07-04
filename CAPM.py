#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Capital Asset Pricing Model (CAPM)
#### Objectives: 
#Use numpy and functions to compute a stock's CAPM beta
#Perform sensitivity analysis to understand how the data points impact the beta estimate

# load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics

# risk-free Treasury rate
R_f = 0.0175 / 252

# read in the market data
data = pd.read_csv('capm_market_data.csv')

'''Look at some records  
SPY is an ETF for the S&P 500 (the "stock market")  
AAPL is Apple  
The values are closing prices, adjusted for splits and dividends'''

df = data.copy()
print(df.head())
print(df.tail())

#drop the date column
df = df.drop(columns = 'date') #remove the date column
df.head()

'''Compute daily returns (percentage changes in price) for SPY, AAPL  
Be sure to drop the first row of NaN'''

returns = df.pct_change(axis=0)
returns.dropna(inplace = True)
returns.head()

#print the first 5 rows of returns
returns.head()

#Save AAPL, SPY returns into separate numpy arrays
SPY, AAPL = returns.spy_adj_close.values, returns.aapl_adj_close.values

#print the five first values from the SPY numpy array and the AAPL numpy array
print(SPY[0:5])
print(AAPL[0:5])

#Compute the excess returns of AAPL, SPY by simply subtracting the constant *R_f* from the returns.
#Specifically, for the numpy array containing AAPL returns, subtract *R_f* from each of the returns. Repeat for SPY returns.
excess_spy = SPY-R_f
excess_aapl = AAPL-R_f

#print the last five excess returns from both AAPL and SPY numpy arrays
print(excess_spy[-5:])
print(excess_aapl[-5:])

#Make a scatterplot with SPY excess returns on x-axis, AAPL excess returns on y axis
plt.scatter(excess_spy,excess_aapl)
plt.xlabel('Stock Market')
plt.ylabel('Apple Stock')
plt.title('Excess Returns of Stock Market against Apple Stock')
plt.grid()
plt.show()

#Use Linear Algebra (matrices) to Compute the Regression Coefficient Estimate betahat
#matrix formula:
#betahat = (x.T*x)^-1*(x.T*y)
#x.T is transpose of x

#reshape SPY and AAPL arrays
x = excess_spy.reshape(-1,1)
y = excess_aapl.reshape(-1,1)
print(np.shape(excess_spy))
print(np.shape(x))
#excess is no longer a one dimensional array

#compute betahat and extract value ([0][0] is the value from the first row and col)
#beta_hat = np.dot(np.linalg.inv(np.dot(x.T,x)),np.dot(x.T,y))[0][0]
#print('Beta is:', round(beta_hat,4))

#alternative way use np.matmul() versus np.dot()
beta_hat2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(), x)), x.transpose()), y)[0][0]
print('Beta is:',beta_hat2)

#Step by step instead of one liner
#xtx = np.matmul(x.transpose(),x)
#xtxi = np.linalg.inv(xtx)
#xtxixt = np.matmul(xtxi,x.transpose())
#beta = np.matmul(xtxixt,y)
#beta_hat = beta[0][0] #take value at row 0, col 0
#print('Beta is:',round(beta_hat,4))

'''Beta estimate is greater than one.  
This means that the risk of AAPL stock, given the data, and according to this particular (flawed) model,  
is higher relative to the risk of the S&P 500.'''

'''Want to understand how sensitive the beta is to each data point.   
Want to drop each data point (one at a time), compute betahat using formula from above, and save each measurement.'''

#write a function beta_sensitivity with specs:
#take numpy arrays x, y
#output a list of tuples. each tuple contains (observation row dropped, beta estimate)
#Hint: **np.delete(x, i).reshape(-1,1)** will delete observation i from array x, and make it a column vector

def beta_sensitivity(x,y):
    out = [] #empty list
    sz = x.shape[0] #for (n,m) where n is # of rows and m # of cols, shape[0] is n
    
    for ix in range(sz):
        xx = np.delete(x,ix).reshape(-1,1) #from array x, remove observation ix
        yy = np.delete(y,ix).reshape(-1,1) #from array y, remove observation ix 
        
        bi = np.matmul(np.matmul(np.linalg.inv(np.matmul(xx.transpose(), xx)), xx.transpose()), yy)[0][0] #oneliner of beta
    
        out.append(bi) #append to out list
        
    return out #returns list of betas sensitive to removal of one observation in each line

#call beta_sensitivity function and print 5 first tuples
beta_sensitivity(x,y)[0:5]

#recall, beta without deletion of one observation is
beta_hat2

##at least for the first five outputs in beta_sensitivity, removing one observation, the change isn't significant
#now look at average for beta_sensitivity
statistics.mean(beta_sensitivity(x,y))
#mean of beta_sensitivity is approximately the same as our beta
