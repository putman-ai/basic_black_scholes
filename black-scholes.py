import yfinance as yf
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    call_price = S * N_d1 - K * np.exp(-r * T) * N_d2
    delta = N_d1
    return call_price, delta

# Download historical stock prices
ticker = "TSLA"
start_date = "2022-10-05"
stock_data = yf.download(ticker, start=start_date)
expiration_days = 50
strike = 250

# Calculate daily returns and volatility
daily_returns = stock_data["Adj Close"].pct_change().dropna()
sigma = daily_returns.std() * np.sqrt(252)

# Calculate theoretical call price
S = stock_data["Adj Close"][-1] # current stock price
K = strike # strike price
T = expiration_days/365 # time in days to expiration (in years)
r = 0.02 # risk-free interest rate
call_price = black_scholes_call(S, K, T, r, sigma)
print("Theoretical call price & delta:", call_price)