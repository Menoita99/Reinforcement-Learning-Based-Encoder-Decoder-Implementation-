import enum
import random
import pandas as pd


class Actions(enum.IntEnum):
    Buy = 0
    Sell = 1
    Noop = 2
    Close = 3

    @staticmethod
    def random():
        rand = random.randint(0, 3)
        if rand == 0:
            return Actions.Buy
        elif rand == 1:
            return Actions.Sell
        elif rand == 2:
            return Actions.Noop
        else:
            return Actions.Close



class Environment:

    def __init__(self, useWindowState=False, windowSize=10, market="EUR_USD", initialMoney=1000,seed=1,dataSrc="data/close ema200 ema50 ema13 ema7 macd_macd macd_signal.csv"):

        self.seed = seed
        self.market = market
        self.useWindowState = useWindowState
        self.windowSize = windowSize
        self.dataSrc = dataSrc

        self.initialMoney = initialMoney
        self.money = initialMoney
        self.minimumMoney = initialMoney * 0.25
        self.leverage = 30

        self.prevMoney = initialMoney
        self.prevAction = Actions.Noop
        self.currentCandle = 0
        self.penalty = 0
        self.position = None

        self.count=0
        self._loadData()


    def _loadData(self):
        self.states = pd.read_csv(self.dataSrc,sep=";").values.tolist()



    def reset(self):
        self.prevAction = Actions.Noop
        self.penalty = 0
        self.count = 0
        self.position = None
        self.money = self.initialMoney
        self.prevMoney = self.initialMoney

        random.seed(self.seed)
        self.currentCandle = random.randint(0 if not self.useWindowState else self.windowSize, int(len(self.states) / 2))
        self.currentPrice = self.states[self.currentCandle][0]

        return self.generateState()



    def addPositionToCandle(self, candle):
        cdl = candle.copy()
        cdl.append(0 if self.position is None else 1 if self.position[0] == Actions.Buy else -1)
        return cdl




    def generateState(self):
        state = self.addPositionToCandle(self.states[self.currentCandle])
        self.currentPrice = state[0]

        if self.useWindowState:
            windowState = []
            for i in range(-self.windowSize+1,0):
                windowState.append(self.addPositionToCandle(self.states[i+self.currentCandle]))
            windowState.append(state)
            return windowState

        return state




    def step(self, action):
        self.currentCandle += 1

        state = self.generateState()

        # print("-----------------------------------------------------------------------------------------------------------------------")
        # print(str(self.getCurrentMoney(self.currentPrice)) + " " + str(self.position))

        self.trade(action, self.currentPrice)

        stop = self.currentCandle + 1 >= len(self.states) or self.getCurrentMoney(self.currentPrice) <= self.minimumMoney

        self.prevAction = action

        output = state,self.reward(action, self.currentPrice), stop, [self.getCurrentMoney( self.currentPrice)]

        # print(str(self.currentCandle) + "-> " + str(action) + "  " + str(output))
        # print(str(self.getCurrentMoney(self.currentPrice)) + " " + str(self.position))
        # if stop:
        #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        if stop:
            print(self.count)
        else:
            self.count += 1
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



    def reward(self, action, price):
        reward = self.getCurrentMoney(price) - self.prevMoney - (.5 if self.position is None else 0) + self.penalty
        self.prevMoney = self.getCurrentMoney(price)
        self.penalty = 0
        return reward



    def trade(self, action, price):
        if action == Actions.Noop:
            return

        if self.position is None:
            if action == Actions.Close:
                self.penalty = -10
                return
                #raise Exception("There is no operation to close")

            self.position = [action, price, self.money * self.leverage, self.money * self.leverage * 0.00001 / price]
        else:
            positionAction, positionPrice, units, pipeteValue = self.position

            if action == positionAction:
                self.penalty = -10
                return
                #raise Exception("Operation have the same action " + str(action))

            elif action == Actions.Close:
                if positionAction == Actions.Buy:
                    pipetes = int((price - positionPrice) * 100_000)
                if positionAction == Actions.Sell:
                    pipetes = int((positionPrice - price) * 100_000)
                self.money = units / self.leverage + pipetes * pipeteValue
                self.position = None
            else:
                if action == Actions.Sell:  # close buy operation
                    pipetes = int((price - positionPrice) * 100_000)
                if action == Actions.Buy:  # close sell operation
                    pipetes = int((positionPrice - price) * 100_000)
                self.money = units / self.leverage + pipetes * pipeteValue
                self.position = [action, price, self.money * self.leverage, self.money * self.leverage * 0.00001 / price]


