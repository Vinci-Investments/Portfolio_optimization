import pandas as pd;
import pandas_datareader.data as web;
import numpy as np;
import matplotlib.pyplot as plt;

# List of stocks composing the portfolio
stocks = ['AAPL', 'FB', 'GOOG', 'MSFT', 'NVDA']


# Gathering historical returns for each stock
data = web.DataReader(stocks,data_source="google",start='01/01/2016')['Close']
returns = data.pct_change()


# CLeaning datas
returns = returns.dropna(axis=0, how='any');


# Daily mean returns, covariance and correlation between stocks
returns_mean = returns.mean()
returns_covariance_matrix = returns.cov()


# Maximum drawdown calculator for a given data_frame
def max_drawdown(portfolio_data_frame, weights, time_period):
    simulated_portfolio = weights[0]*portfolio_data_frame.ix[:,0]
    for i in range(1, len(portfolio_data_frame.columns)):
        simulated_portfolio += weights[i]*portfolio_data_frame.ix[:,i]
    max_drawdown_value = float('-inf')
    for i in range(int(len(simulated_portfolio)/time_period)-1):
        biggest_variation = max(simulated_portfolio[i*time_period:(i+1)*time_period])/min(simulated_portfolio[i*time_period:(i+1)*time_period])
        if(biggest_variation > max_drawdown_value):
            max_drawdown_value = biggest_variation
    return max_drawdown_value


# Parameter for the Monte-Carlo simulation
nbr_iteration = 4000
simulated_portfolios = np.zeros((4, nbr_iteration)) # mean, std, Sharpe ratio and max drawdown
simulated_weights = []
risk_free_rate = 0.01 # Livret A
nbr_trading_days = 252
time_period_drawdown = 20
    

for index in range(nbr_iteration):
    
    # Randomly creating the array of weight and then normalizing such that the sum equals 1
    weights = np.random.rand(1, len(stocks))[0]
    weights = weights/np.sum(weights)
    simulated_weights.append(weights)
    
    # Computing the return and volatility of the portfolio with those weights
    portfolio_return = np.sum(returns_mean.values * weights * nbr_trading_days)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_covariance_matrix.values, weights))* nbr_trading_days**2)

    # Store results of the simulation
    simulated_portfolios[0, index] = portfolio_return 
    simulated_portfolios[1, index] = portfolio_volatility
    simulated_portfolios[2, index] = (portfolio_return - risk_free_rate) / portfolio_volatility
    simulated_portfolios[3, index] = max_drawdown(data, weights, time_period_drawdown)


# Initializing the figure to plot
figure, axarr = plt.subplots(1,2)


# Diplaying results of the simulation
simulated_portfolios_df = pd.DataFrame(simulated_portfolios.T,columns=['retrn','stdv','sharpe', 'max_drdwn'])
scat = axarr[0].scatter(simulated_portfolios_df.stdv,simulated_portfolios_df.retrn
            ,c=simulated_portfolios_df.sharpe,cmap='RdYlBu')
figure.colorbar(scat, ax = axarr[0])
axarr[0].set_title("Simulated portfolios")


# Finding the maximum Sharpe ratio, minimum volatility and minimum drawdown
highest_sharpe_position = simulated_portfolios_df['sharpe'].idxmax()
highest_sharpe = simulated_portfolios_df.iloc[highest_sharpe_position]
highest_sharpe_weights = simulated_weights[highest_sharpe_position]
lowest_volatility_position = simulated_portfolios_df['stdv'].idxmin()
lowest_volatility = simulated_portfolios_df.iloc[lowest_volatility_position]
lowest_volatility_weights = simulated_weights[lowest_volatility_position]
lowest_drawdown_position = simulated_portfolios_df['max_drdwn'].idxmin()
lowest_drawdown = simulated_portfolios_df.iloc[lowest_drawdown_position]
lowest_drawdown_weights = simulated_weights[lowest_drawdown_position]


# Plotting the three optimal points on the efficient curve
axarr[0].scatter(highest_sharpe[1],highest_sharpe[0],marker=(4,0,0),color='b',s=400)
axarr[0].scatter(lowest_volatility[1],lowest_volatility[0],marker=(4,0,0),color='g',s=400)
axarr[0].scatter(lowest_drawdown[1],lowest_drawdown[0],marker=(4,0,0),color='r',s=400)


# Computing the optimized portfolios returns
data_sharpe_weighted = highest_sharpe_weights[0]*data.ix[:,0]
data_equally_weighted = (1/len(stocks))*data.ix[:,0]
data_volatility_weighted = lowest_volatility_weights[0]*data.ix[:,0]
data_drawdown_weighted = lowest_drawdown_weights[0]*data.ix[:,0]

for i in range(1, len(stocks)):
    data_sharpe_weighted += highest_sharpe_weights[i]*data.ix[:,i]
    data_equally_weighted += (1/len(stocks))*data.ix[:,i]
    data_volatility_weighted += lowest_volatility_weights[i]*data.ix[:,i]
    data_drawdown_weighted += lowest_drawdown_weights[i]*data.ix[:,i]


# Normalizing
data_sharpe_weighted = data_sharpe_weighted/data_sharpe_weighted[0]
data_equally_weighted = data_equally_weighted/data_equally_weighted[0]
data_volatility_weighted = data_volatility_weighted/data_volatility_weighted[0]
data_drawdown_weighted = data_drawdown_weighted/data_drawdown_weighted[0]


# Plotting the optimized portfolios returns
axarr[1].plot(data_sharpe_weighted, label="Sharpe optimization")
axarr[1].plot(data_equally_weighted, label="Equally weighted")
axarr[1].plot(data_volatility_weighted, label="Volatility optimization")
axarr[1].plot(data_drawdown_weighted, label="Drawdown optimization")
axarr[1].legend()
axarr[1].set_title("Returns comparaison")
plt.show()
    
    






