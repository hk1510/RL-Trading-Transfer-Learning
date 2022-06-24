import gym
from gym import spaces
import numpy as np
import pandas_datareader as pdr
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
	
class TradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, marketSymbol='^DJI', startingDate='2012-1-1', endingDate='2018-1-1', money=100000, transactionCosts=0, startingPoint=0):
        super(TradingEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3,)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.money = money
        self.transactionCosts = transactionCosts
        self.marketSymbol = marketSymbol
        self.startingPoint = startingPoint
        self.e = 0
        

    def step(self, action):
        action = action - 1
        self.data['Action'][self.t] = action
        if action > 0:
            # agent buys
            num_shares = 0
            buy_amount = self.data['Cash'][self.t-1] * action
            
            if (buy_amount / self.data['Open'][self.t]  >= 1):
                num_shares = int(buy_amount / self.data['Open'][self.t])

            self.data['Holdings'][self.t] = self.data['Holdings'][self.t - 1] + num_shares
            self.data['Cash'][self.t] = self.data['Cash'][self.t - 1] - num_shares * self.data['Open'][self.t]


        elif action == 0:
            # agent holds
            self.data['Holdings'][self.t] = self.data['Holdings'][self.t - 1]
            self.data['Cash'][self.t] = self.data['Cash'][self.t - 1]

        else:
            # agent sells
            num_shares_sold = int(self.data['Holdings'][self.t - 1] * abs(action))
            self.data['Holdings'][self.t] = self.data['Holdings'][self.t - 1] - num_shares_sold
            self.data['Cash'][self.t] = self.data['Cash'][self.t - 1] + num_shares_sold * self.data['Open'][self.t]

        self.data['Money'][self.t] = self.data['Holdings'][self.t] * self.data['Open'][self.t] + self.data['Cash'][self.t]
        self.data['Returns'][self.t] = (self.data['Money'][self.t] - self.data['Money'][self.t - 1]) / self.data['Money'][self.t - 1]

        observation = np.array([self.data['Open'][self.t], 
                self.data['Close'][self.t],
                self.data['Low'][self.t],
                self.data['High'][self.t],
                self.data['Volume'][self.t],
                action])

        reward = self.data['Returns'][self.t]

        if (self.t + 1 >= len(self.data)):
            print("Portfolio Value at Epoch " + str(self.e) + ": " + str(self.data['Money'][self.t]))

        self.t = self.t + 1

        done = False
        if self.t >= len(self.data):
            done = True
            self.e = self.e + 1

        info = {}

        return observation, reward, done, info

    def reset(self):
        stock = self.marketSymbol

        try:
            self.data = pd.read_csv('Data/' + stock + '--' + self.startingDate + '--' + self.endingDate + '.csv')
            print('Read from dataframe')
        except:
            self.data = pdr.data.DataReader(stock, 'yahoo', self.startingDate, self.endingDate)
            self.data.to_csv('Data/' + stock + '--' + self.startingDate + '--' + self.endingDate + '.csv')
            print('Read from pandas datareader')

        self.data = self.processDataframe(self.data)

        # Interpolate missing data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)

        # Set the trading activity dataframe
        self.data['Action'] = 0.
        self.data['Holdings'] = 0
        self.data['Cash'] = float(self.money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        self.t = 1

        observation = np.array([self.data['Open'][0], 
                       self.data['Close'][0],
                       self.data['Low'][0],
                       self.data['High'][0],
                       self.data['Volume'][0],
                       0])

        return observation  # reward, done, info can't be included

    def processDataframe(self, dataframe):
        """
        GOAL: Process a downloaded dataframe to homogenize the output format.
        
        INPUTS:     - dataframe: Pandas dataframe to be processed.
          
        OUTPUTS:    - dataframe: Processed Pandas dataframe.
        """
        
        # Remove useless columns
        dataframe['Close'] = dataframe['Adj Close']
        del dataframe['Adj Close']
        
        # Adapt the dataframe index and column names
        dataframe.index.names = ['Timestamp']
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]

        return dataframe

    def render(self, mode='human'):
        return self.data


class TradingTransferEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, marketSymbols=['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'FB', 'NVDA', 'BRK-B', 'JPM'], runsPerStock=2, startingDate='2012-1-1', endingDate='2018-1-1', money=100000, transactionCosts=0, startingPoint=0):
        super(TradingTransferEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3,)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.money = money
        self.transactionCosts = transactionCosts
        self.marketSymbols = marketSymbols
        self.startingPoint = startingPoint
        self.e = 0
        self.runsPerStock = runsPerStock
        

    def step(self, action):
        action = action - 1
        self.data['Action'][self.t] = action
        if action > 0:
            # agent buys
            num_shares = 0
            buy_amount = self.data['Cash'][self.t-1] * action
            
            if (buy_amount / self.data['Open'][self.t]  >= 1):
                num_shares = int(buy_amount / self.data['Open'][self.t])

            self.data['Holdings'][self.t] = self.data['Holdings'][self.t - 1] + num_shares
            self.data['Cash'][self.t] = self.data['Cash'][self.t - 1] - num_shares * self.data['Open'][self.t]


        elif action == 0:
            # agent holds
            self.data['Holdings'][self.t] = self.data['Holdings'][self.t - 1]
            self.data['Cash'][self.t] = self.data['Cash'][self.t - 1]

        else:
            # agent sells
            num_shares_sold = int(self.data['Holdings'][self.t - 1] * abs(action))
            self.data['Holdings'][self.t] = self.data['Holdings'][self.t - 1] - num_shares_sold
            self.data['Cash'][self.t] = self.data['Cash'][self.t - 1] + num_shares_sold * self.data['Open'][self.t]

        self.data['Money'][self.t] = self.data['Holdings'][self.t] * self.data['Open'][self.t] + self.data['Cash'][self.t]
        self.data['Returns'][self.t] = (self.data['Money'][self.t] - self.data['Money'][self.t - 1]) / self.data['Money'][self.t - 1]

        observation = np.array([self.data['Open'][self.t], 
                self.data['Close'][self.t],
                self.data['Low'][self.t],
                self.data['High'][self.t],
                self.data['Volume'][self.t],
                action])

        reward = self.data['Returns'][self.t]

        if (self.t + 1 >= len(self.data)):
            print("Portfolio Value at Epoch " + str(self.e) + ": " + str(self.data['Money'][self.t]))

        self.t = self.t + 1

        done = False
        if self.t >= len(self.data):
            done = True
            self.e = self.e + 1

        info = {}

        return observation, reward, done, info

    def reset(self):
        stocks = self.marketSymbols
        stock = '^DJI'
        if (int(self.e / self.runsPerStock) < len(stocks)):
            stock = self.marketSymbols[int(self.e / self.runsPerStock)]
        print("Stock being trained on at epoch " + str(self.e) + ": " + stock)
        try:
            self.data = pd.read_csv('Data/' + stock + '--' + self.startingDate + '--' + self.endingDate + '.csv')
            print('Read from dataframe')
        except:
            self.data = pdr.data.DataReader(stock, 'yahoo', self.startingDate, self.endingDate)
            self.data.to_csv('Data/' + stock + '--' + self.startingDate + '--' + self.endingDate + '.csv')
            print('Read from pandas datareader')
        
        
        self.data = self.processDataframe(self.data)

        # Interpolate missing data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)

        # Set the trading activity dataframe
        self.data['Action'] = 0.
        self.data['Holdings'] = 0
        self.data['Cash'] = float(self.money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        self.t = 1

        observation = np.array([self.data['Open'][0], 
                       self.data['Close'][0],
                       self.data['Low'][0],
                       self.data['High'][0],
                       self.data['Volume'][0],
                       0])

        return observation  # reward, done, info can't be included

    def processDataframe(self, dataframe):
        """
        GOAL: Process a downloaded dataframe to homogenize the output format.
        
        INPUTS:     - dataframe: Pandas dataframe to be processed.
          
        OUTPUTS:    - dataframe: Processed Pandas dataframe.
        """
        
        # Remove useless columns
        dataframe['Close'] = dataframe['Adj Close']
        del dataframe['Adj Close']
        
        # Adapt the dataframe index and column names
        dataframe.index.names = ['Timestamp']
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]

        return dataframe

    def render(self, mode='ansi'):
        return "Return at time step", self.t, ":", self.data['Returns'][self.t]

class TradingEvalEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, marketSymbol='^DJI', startingDate='2018-1-1', endingDate='2022-3-1', money=100000, transactionCosts=0, startingPoint=0):
        super(TradingEvalEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3,)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.money = money
        self.transactionCosts = transactionCosts
        self.marketSymbol = marketSymbol
        self.startingPoint = startingPoint
        

    def step(self, action):
        action = action - 1
        self.data['Action'][self.t] = action
        if action > 0:
            # agent buys
            num_shares = 0
            buy_amount = self.data['Cash'][self.t-1] * action
            
            if (buy_amount / self.data['Open'][self.t]  >= 1):
                num_shares = int(buy_amount / self.data['Open'][self.t])

            self.data['Holdings'][self.t] = self.data['Holdings'][self.t - 1] + num_shares
            self.data['Cash'][self.t] = self.data['Cash'][self.t - 1] - num_shares * self.data['Open'][self.t]


        elif action == 0:
            # agent holds
            self.data['Holdings'][self.t] = self.data['Holdings'][self.t - 1]
            self.data['Cash'][self.t] = self.data['Cash'][self.t - 1]

        else:
            # agent sells
            num_shares_sold = int(self.data['Holdings'][self.t - 1] * abs(action))
            self.data['Holdings'][self.t] = self.data['Holdings'][self.t - 1] - num_shares_sold
            self.data['Cash'][self.t] = self.data['Cash'][self.t - 1] + num_shares_sold * self.data['Open'][self.t]

        self.data['Money'][self.t] = self.data['Holdings'][self.t] * self.data['Open'][self.t] + self.data['Cash'][self.t]
        self.data['Returns'][self.t] = (self.data['Money'][self.t] - self.data['Money'][self.t - 1]) / self.data['Money'][self.t - 1]

        observation = np.array([self.data['Open'][self.t], 
                self.data['Close'][self.t],
                self.data['Low'][self.t],
                self.data['High'][self.t],
                self.data['Volume'][self.t],
                action])

        reward = self.data['Returns'][self.t]

        self.t = self.t + 1

        done = False
        if self.t >= len(self.data):
            done = True

        info = {}

        return observation, reward, done, info

    def reset(self):
        spy_top_10 = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'FB', 'NVDA', 'BRK-B', 'JPM']
        stock = self.marketSymbol
        try:
            self.data = pd.read_csv('Data/' + stock + '--' + self.startingDate + '--' + self.endingDate + '.csv')
            print('Read from dataframe')
        except:
            self.data = pdr.data.DataReader(stock, 'yahoo', self.startingDate, self.endingDate)
            self.data = self.processDataframe(self.data)

            # Interpolate missing data
            self.data.replace(0.0, np.nan, inplace=True)
            self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
            self.data.fillna(method='ffill', inplace=True)
            self.data.fillna(method='bfill', inplace=True)
            self.data.fillna(0, inplace=True)

            self.data.reset_index(inplace=True)
            self.data.to_csv('Data/' + stock + '--' + self.startingDate + '--' + self.endingDate + '.csv')
            print('Read from pandas datareader')


        # Set the trading activity dataframe
        self.data['Action'] = 0.
        self.data['Holdings'] = 0
        self.data['Cash'] = float(self.money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        self.t = 1

        observation = np.array([self.data['Open'][0], 
                       self.data['Close'][0],
                       self.data['Low'][0],
                       self.data['High'][0],
                       self.data['Volume'][0],
                       0])

        return observation  # reward, done, info can't be included

    def processDataframe(self, dataframe):
        """
        GOAL: Process a downloaded dataframe to homogenize the output format.
        
        INPUTS:     - dataframe: Pandas dataframe to be processed.
          
        OUTPUTS:    - dataframe: Processed Pandas dataframe.
        """
        
        # Remove useless columns
        dataframe['Close'] = dataframe['Adj Close']
        del dataframe['Adj Close']
        
        # Adapt the dataframe index and column names
        dataframe.index.names = ['Timestamp']
        dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']]

        return dataframe

    def render(self, mode='human'):
        return self.data