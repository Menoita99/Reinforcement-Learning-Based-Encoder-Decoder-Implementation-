import enum
import configparser
from mysql import connector
from collections import deque
from random import randint


class Environment:

    def __init__(self, useWindowState=False, windowSize=4, market="EUR_USD", timeframe="H12", train=True):
        self.market = market
        self.timeframe = timeframe
        self.useWindowState = useWindowState
        self.windowSize = windowSize
        self.ownShare = False
        self.currentCandle = 0

        if self.useWindowState:
            self.candles = deque([None] * self.windowSize, maxlen=windowSize)
        self._loadData(train)

        self.actualAction = Actions.Sell
        self.price1 = self.data[0][3]

    def _loadData(self, train):
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
        if (train):
            self.data = self.data[:int(len(self.data) * 0.8)]
        else:
            self.data = self.data[int(len(self.data) * 0.2):]

    def reset(self):
        self.ownShare = False
        self.currentCandle = 0
        if self.useWindowState:
            self.candles = [None] * self.windowSize
        return self.initState(self)

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

        output = state, self.reward(action), self.currentCandle >= len(self.data), "TODO info"
        self.currentCandle += 1
        return output

    """
        Here we should use ask price and bid price to simulate the real world trade application
    """

    def reward(self, action):
        price2 = self.data[self.currentCandle][3]
        if self.actualAction != action and action != Actions.Noop:
            self.price1 = price2

        if action == Actions.Buy or (action == Actions.Noop and self.ownShare):
            reward = ((price2 / self.price1) - 1) * 100
            self.actualAction = action
            return reward
        else:
            reward = ((self.price1 / price2) - 1) * 100
            self.actualAction = action
            return reward


class Actions(enum.Enum):
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


env = Environment(useWindowState=True)
print(env.initState())
for _ in range(100):
    print(env.step(Actions.Noop))
