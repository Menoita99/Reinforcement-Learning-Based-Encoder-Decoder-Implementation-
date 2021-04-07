import enum


# Using enum class create enumerations


class Actions(enum.Enum):
    Buy = 0
    Sell = 1
    Noop = 2

    @staticmethod
    def random():
        #TODO
        return Actions.Buy


class Environment:

    def __init__(self, windowState=False, market="EUR/USD", timeframe="D", iniCountBalance=1000):
        self.iniCountBalance = iniCountBalance
        self.market = market
        self.timeframe = timeframe
        self.windowState = windowState
        self.currentTransaction = None
        self.currentCandle = None
        self.candleIterator = []  #TODO Load with pythorch from db
        if (windowState):
            self.candles = []

    def reset(self):
        #TODO
        return

    def initState(self):
        # TODO
        return (0,0,0,0)

    def step(self,action):
        # TODO
        print("step")

    """
        Here we should use ask price and bid price to simulate the real world trade application
    """
    def _reward(self,action):
        # TODO implement reward function
        return 0
