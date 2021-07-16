import enum
import configparser
from mysql import connector
from collections import deque
from random import randint
import pandas as pd
import numpy as np


class Environment:

    def __init__(self, useWindowState=False, windowSize=4, market="EUR_USD", timeframe="H1", train=True,
                 initialMoney=1000):
        self.market = market
        self.timeframe = timeframe
        self.useWindowState = useWindowState
        self.windowSize = windowSize
        self.currentCandle = 0
        self.position = None

        if self.useWindowState:
            self.candles = deque([None] * self.windowSize, maxlen=self.windowSize)
        self._loadData(market, train)

        self.price1 = self.data[0][3]

        self.initialMoney = initialMoney
        self.money = initialMoney
        self.minimumMoney = initialMoney * 0.25
        self.prevMoney = initialMoney

        self.prevAction = Actions.Noop
        self.leverage = 30

        self.poistionLiveTime = 0




    def _loadData(self, market, train):
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
            select = "SELECT openPercentage,highPercentage,lowPercentage, closePercentage " \
                     "FROM forexdb.candlestick " \
                     "where market='{}' and timeframe='{}' and datetime > '2015-01-01 00:00:00'" \
                     "order by datetime asc;".format(self.market, self.timeframe)

            mycursor.execute(select)
            self.data = mycursor.fetchall()

            select = "SELECT close " \
                     "FROM forexdb.candlestick " \
                     "where market='{}' and timeframe='{}' and  datetime > '2015-01-01 00:00:00' " \
                     "order by datetime asc limit 1;".format(self.market, self.timeframe)

            mycursor.execute(select)
            self.startPrice = mycursor.fetchone()[0]
            self.currentPrice = self.startPrice
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
        self.currentCandle = 0
        self.position = None
        self.money = self.initialMoney
        self.prevMoney = self.initialMoney
        self.currentPrice = self.startPrice
        if self.useWindowState:
            self.candles = deque([None] * self.windowSize, maxlen=self.windowSize)
        return self.initState()




    def initState(self):
        if self.useWindowState:
            output = [None] * self.windowSize
            for i in range(self.windowSize):
                self.candles.popleft()
                self.candles.append(self.addPositionToCandle(self.data[i]))
            self.currentCandle = self.windowSize
            return output
        else:
            self.currentCandle = 1
            return self.addPositionToCandle(self.data[0])




    def addPositionToCandle(self, candle):
        if (self.position == None):
            return candle + (0,)
        else:
            return candle + ((1,) if self.position[0] == Actions.Buy else (-1,))




    def step(self, action):
        state = self.data[self.currentCandle]

        if self.useWindowState:
            self.candles.popleft()
            self.candles.append(state)
            state = self.candles

        # print("-----------------------------------------------------------------------------------------------------------------------")
        # print(str(self.getCurrentMoney(self.currentPrice)) + " " + str(self.position))

        self.currentPrice = round((state[3] / 100 + 1) * self.currentPrice, 5)

        try:
            self.trade(action, self.currentPrice)
            stop = self.currentCandle + 1 >= len(self.data) or self.getCurrentMoney(self.currentPrice) <= self.minimumMoney
        except:
            stop = True

        self.prevAction = action

        state = self.addPositionToCandle(state)
        output = state,self.reward(action, self.currentPrice), stop, [self.getCurrentMoney( self.currentPrice)]

        # print(str(self.currentCandle) + "-> " + str(action) + "  " + str(output))
        # print(str(self.getCurrentMoney(self.currentPrice)) + " " + str(self.position))
        # if stop:
        #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        self.poistionLiveTime += 1
        self.currentCandle += 1
        return output



    def getCurrentMoney(self,price):
        if self.position is None:
            return self.money
        else:
            positionAction, positionPrice, units, pipeteValue = self.position
            if positionAction == Actions.Buy:
                pipetes = int((price - positionPrice) * 100_000)
            if positionAction == Actions.Sell:
                pipetes = int((positionPrice - price) * 100_000)
            return units / self.leverage + pipetes * pipeteValue



    """
        Here we should use ask price and bid price to simulate the real world trade application
    """
    def reward(self, action, price):
        reward = self.getCurrentMoney(price) - self.prevMoney
        self.prevMoney = self.getCurrentMoney(price)
        return reward - (.5 if self.position is None else 0)#(10 * self.poistionLiveTime if self.position is None else -1)




    def trade(self, action, price):
        if action == Actions.Noop:
            return

        if self.position is None:
            if action == Actions.Close:
                raise Exception("There is no operation to close")
            self.position = [action, price, self.money * self.leverage, self.money * self.leverage * 0.00001 / price]
            self.poistionLiveTime = 0
        else:
            positionAction, positionPrice, units, pipeteValue = self.position
            if action == positionAction:
                raise Exception("Operation have the same action " + str(action))
            elif action == Actions.Close:
                if positionAction == Actions.Buy:
                    pipetes = int((price - positionPrice) * 100_000)
                if positionAction == Actions.Sell:
                    pipetes = int((positionPrice - price) * 100_000)
                self.money = units / self.leverage + pipetes * pipeteValue
                self.position = None
                self.poistionLiveTime = 0
            else:
                if action == Actions.Sell:  # close buy operation
                    pipetes = int((price - positionPrice) * 100_000)
                if action == Actions.Buy:  # close sell operation
                    pipetes = int((positionPrice - price) * 100_000)
                self.money = units / self.leverage + pipetes * pipeteValue
                self.position = [action, price, self.money * self.leverage, self.money * self.leverage * 0.00001 / price]
                self.poistionLiveTime = 0



class Actions(enum.IntEnum):
    Buy = 0
    Sell = 1
    Noop = 2
    Close = 3

    @staticmethod
    def random():
        rand = randint(0, 3)
        if rand == 0:
            return Actions.Buy
        elif rand == 1:
            return Actions.Sell
        elif rand == 2:
            return Actions.Noop
        else:
            return Actions.Close
