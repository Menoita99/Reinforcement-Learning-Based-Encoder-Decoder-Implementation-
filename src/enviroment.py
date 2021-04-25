import enum
import configparser
from mysql import connector
from collections import deque
from random import randint
import pandas as pd
import numpy as np

class Environment:

    def __init__(self, useWindowState=False, windowSize=4, market="BTC_USD", timeframe="D",train=True,initialMoney=1000):
        self.market = market
        self.timeframe = timeframe
        self.useWindowState = useWindowState
        self.windowSize = windowSize
        self.ownShare = False
        self.currentCandle = 0

        if self.useWindowState:
            self.candles = deque([None] * self.windowSize, maxlen=self.windowSize)
        self._loadData(market,train)

        self.prevAction = Actions.Noop
        self.price1 = self.data[0][3]
        self.transaction = [Actions.Sell, 0]
        self.initialMoney = initialMoney
        self.money = self.initialMoney



    def _loadData(self,market, train):
        if market == "EUR_USD":
            config = configparser.RawConfigParser()
            config.read('application.properties')

            mydb = connector.connect(
                host=config.get('Database', 'database.host'),
                user=config.get('Database', 'database.user'),
                password=config.get('Database', 'database.password'),
                database=config.get('Database', 'database.db')
            )
            mycursor = mydb.cursor()
            select = "SELECT open,high,low, close FROM forexdb.candlestick where market='{}' and timeframe='{}' order by datetime asc;".format(
                self.market, self.timeframe)

            mycursor.execute(select)
            self.data = mycursor.fetchall()

        else:
            df = pd.read_csv(r'data\{}.csv'.format(market))
            df = df.dropna(how='any', axis=0)
            df = df[['Open', 'High', 'Low', 'Close']]
            df = df.astype(np.float32)

            self.data = []
            for index, rows in df.iterrows():
                row = [rows.Open, rows.High, rows.Low, rows.Close]
                self.data.append(row)


        self.data = self.data[:int(len(self.data) * 0.8)] if train else self.data[int(len(self.data) * 0.2):]


    def reset(self):
        self.ownShare = False
        self.currentCandle = 0
        self.money = self.initialMoney
        if self.useWindowState:
            self.candles = deque([None] * self.windowSize, maxlen=self.windowSize)
        return self.initState()


    def initState(self):
        if self.useWindowState:
            output = [None] * self.windowSize
            for i in range(self.windowSize):
                output[i] = self.data[i]
                self.candles.popleft()
                self.candles.append(self.data[i])
            self.currentCandle = self.windowSize
            return output
        else:
            self.currentCandle = 1
            return self.data[0]

    def step(self, action):
        state = self.data[self.currentCandle]
        if self.useWindowState:
            self.candles.popleft()
            self.candles.append(state)
            state = self.candles

        price = self.data[self.currentCandle][3]
        self.trade(action,price)
        currentMoney = self.money if self.transaction[1] == 0 else self.transaction[1] * price

        output = state, self.reward(action), self.currentCandle+1 >= len(self.data), [currentMoney]
        self.currentCandle += 1
        return output



    """
        Here we should use ask price and bid price to simulate the real world trade application
    """
    def reward(self, action):
        price2 = self.data[self.currentCandle][3]
        if self.prevAction != action and action != Actions.Noop:
            self.price1 = price2

        if action == Actions.Buy or (action == Actions.Noop and self.ownShare):
            reward = ((price2/self.price1) - 1)*100 - 1  # -1 to simulate tax
            self.prevAction = action
            self.ownShare = True
            return reward
        else:
            reward = ((self.price1/price2) - 1)*100 - 1  # -1 to simulate tax
            self.prevAction = action
            self.ownShare = False
            return reward



    def trade(self, action, price):
        prevAction, units = self.transaction
        if prevAction == Actions.Sell and action == Actions.Buy:
            self.transaction = [action,self.money/price]
        elif prevAction == Actions.Buy and action == Actions.Sell:
            self.money = price * units * 0.99
            self.transaction = [action, 0]



class Actions(enum.IntEnum):
    Buy = 0
    Sell = 1
    Noop = 2

    @staticmethod
    def random():
        rand = randint(0, 2)
        if rand == 0:
            return Actions.Buy
        elif rand == 1:
            return Actions.Sell
        else:
            return Actions.Noop
