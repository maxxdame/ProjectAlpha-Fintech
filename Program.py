#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from utils import read_company_profile
from main import banana
import pandas as pd
from datetime import date as dt
from pandas_datareader import data
from random import choice
import matplotlib.pyplot as plt 
import numpy as np
import bt 
import math

"""
Created on Sat Feb 15 09:34:36 2020
@author: Christopher
"""


"""
GET DATAFRAME WITH UNDERVALUED COMPANIES FROM MAIN.PY (BANANA) AND CONVERT IT TO 
    LIST WHICH IS THEN USED AS SOURCE FOR "TICKERS" FEEDING INTO ALL THE OTHER RATIOS
"""

ticker_list = banana
ticker_list.columns = ["Ticker"]


"""
#GENERAL INPUTS FOR FOLLOWING CODE
"""

tickers = ticker_list["Ticker"].to_list()
start_date = dt(2015,1,1) #2015 for current portfolio and 2010 for backtesting portfolio
end_date = dt(2020, 2, 17)

RFR = 0.018 #annual TBOND (10y(1.34%) or 30y?) yield bc we look for long term

number_of_simulations = 2500

"""
#OPTIMIZE FOR MAX SHARPE RATIO BY SOURCING TICKERS FROM DCF OUTPUT
"""

table = bt.get(tickers, start = start_date, end = end_date)

# calculate daily and annual returns of the stocks
returns_daily = table.pct_change().dropna()
returns_annual = returns_daily.mean() * 252

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 252

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(tickers)
num_portfolios = number_of_simulations

#set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    excess_returns = np.dot(weights, returns_annual) - RFR
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = excess_returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(excess_returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Excess Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(tickers):
    portfolio[symbol] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Excess Returns', 'Volatility', 'Sharpe Ratio'] + [stock for stock in tickers]

# reorder dataframe columns
df = df[column_order]

# find min Volatility & max sharpe values in the dataframe (df)
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]
sharpe_ratio = round(float(sharpe_portfolio["Sharpe Ratio"]), 3)

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Excess Returns', c='Sharpe Ratio', cmap='Blues', edgecolors='none', figsize=(10, 8), grid=True, marker = 'p')
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Excess Returns'], c='red', marker='+', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Excess Returns'], c='blue', marker='+', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Excess Returns')
plt.title('Efficient Frontier')
plt.savefig('Efficient_frontier_max_port.png')


"""
#VALUE AT RISK (VaR) USING TICKERS FROM DCF OUTPUT
"""

def get_Portfolio(tickers, start_date, end_date):
    stock_data = data.DataReader(tickers, data_source='yahoo', start = start_date, end = end_date)['Adj Close']
    
    return stock_data

# Change shape of DataFrame with optimized weights so they can be used as a source for VaR
sharpe_port_weights = sharpe_portfolio.drop(columns =["Excess Returns", "Volatility", "Sharpe Ratio"]).transpose()
sharpe_port_weights.columns = ["Weight"]

# Set up Weights
weights = pd.Series(index = tickers, dtype = float)
weights[tickers]= sharpe_port_weights["Weight"]

# Monte Carlo paramters
monte_carlo_runs = number_of_simulations
days_to_simulate = 5
loss_cutoff = 0.99 #count any losses larger than X% 

# Call that simple function we wrote above
what_we_got = get_Portfolio(tickers, start_date, end_date)

# Compute returns from those Adjusted Closes
returns = what_we_got[tickers].pct_change()
returns = returns.dropna() 

# Calculate mu and sigma
mu = returns.mean()
sigma= returns.std()

# Monte Carlo VaR loop
compound_returns = sigma.copy()
total_simulations = 0
bad_simulations = 0
for run_counter in range(0,monte_carlo_runs): # Loop over runs    
    for i in tickers: # loop over tickers
        # Loop over simulated days:
        compounded_temp = 1
        for simulated_day_counter in range(0,days_to_simulate): # loop over days
            simulated_return = choice(returns[i])
            compounded_temp = compounded_temp * (simulated_return + 1)        
        compound_returns[i]=compounded_temp # store compounded returns
    # Now see if those returns are bad by combining with weights
    portfolio_return = compound_returns.dot(weights) # dot product
    if(portfolio_return<loss_cutoff):
        bad_simulations = bad_simulations + 1
    total_simulations = total_simulations + 1


"""
#TREYNOR
"""
#Calculate beta by multiplying every company's beta by it's portfolio weight
def beta(comp):
    beta = read_company_profile(comp)
    
    ß = pd.DataFrame([float(beta["beta"])])
    return ß
#Create DataFrame with every company's beta
ß_df = pd.DataFrame()
for comp in tickers: 
    t = beta(comp)
    ß_df = ß_df.append(t)

#Combine every company's ß and weights to calculate portfolio ß
ß_df.columns = ["Beta"]
ß_df.index = tickers
weights_TT = pd.DataFrame([sharpe_port_weights["Weight"]]).transpose()
ß_dff = pd.concat([weights_TT, ß_df], axis = 1)
ß_dff.columns = ["Weight", "Beta"]
ß_dff["Components Portfolio Beta"] = ß_dff["Beta"] * ß_dff["Weight"]
ß_T = sum(ß_dff["Components Portfolio Beta"])

treynor_ratio = round(float((sharpe_portfolio["Excess Returns"]) / ß_T), 3)


"""
#SORTINO RATIO
"""

#S&P500 for r (Benchmark)
Benchmark = ["^GSPC"]
SnP = bt.get(Benchmark, start = start_date, end = end_date)
returns_daily_SnP = SnP.pct_change().dropna()

#Portfolio returns for mu and  Benchmark for r
mu = float(sharpe_portfolio["Excess Returns"]) + RFR
r = float(returns_daily_SnP.mean() * 252)
r_d = float(returns_daily_SnP.mean())


#Create DataFrame with every company's weighted return and the sum them up for every period
returns = pd.DataFrame() 
for i in tickers:
    bb = table[i.lower()].pct_change().dropna() * float(sharpe_portfolio[i])
    returns = pd.concat([returns,bb], axis = 1)

returns["Returns"] = returns.sum(axis = 1)
returns_portfolio = returns["Returns"]

#Calculate downside risk
rp_downside = pd.DataFrame(returns_portfolio)
semi = pd.Series([])
for i in range(len(rp_downside["Returns"])):
    if rp_downside["Returns"][i] - r_d > 0:
        semi[i] = 0
    elif rp_downside["Returns"][i] - r_d == 0:
        semi[i] = 0
    else:
        semi[i] = (float(rp_downside["Returns"][i] - r_d)**2)


#Sortino Calculation
excess_sortino = float(mu - r)
downside = math.sqrt(semi.mean()) * math.sqrt(252)

sortino_ratio = excess_sortino / downside


"""
#INFORMATION RATIO (average alpha per unit of risk)
"""

returns_benchmark = returns_daily_SnP["gspc"].copy()
combo = pd.concat([returns_portfolio, returns_benchmark], axis = 1)
combo.columns = ["Port Returns", "Benchmark Returns"]

excess_return = returns_portfolio - returns_daily_SnP["gspc"]

rp = mu
rb = r
er = excess_return.std() * math.sqrt(252)

information_ratio = (rp - rb) / er

    
"""
#OUTPUTS
"""
#print(round(weights_TT * 100, 2))

print("The Sharpe Ratio is ", sharpe_ratio,".")
print("The Treynor Ratio is ", treynor_ratio,".")
print("VaR: The portfolio lost",round((1-loss_cutoff)*100,3),"%", "over",days_to_simulate,"days", round(bad_simulations/total_simulations * 100, 2), "% of the time")
print("The Sortino Ratio is ",round(sortino_ratio, 3))
print("The Information Ratio is ",round(information_ratio, 3))
