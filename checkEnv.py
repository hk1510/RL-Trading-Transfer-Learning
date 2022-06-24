from stable_baselines3.common.env_checker import check_env
from env import TradingEnv

startingDate = '2012-1-1'
endingDate = '2020-1-1'
splitingDate = '2018-1-1'
stock = 'AAPL'
money = 100000
stateLength = 30
percentageCosts = [0, 0.1, 0.2]
transactionCosts = percentageCosts[1]/100

env = TradingEnv()

check_env(env)