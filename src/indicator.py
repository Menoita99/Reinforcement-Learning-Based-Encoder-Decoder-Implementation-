class Ema:
    """
    EMA=Price(t)×k+EMA(y)×(1−k)
    where:
    t=today
    y=yesterday
    N=number of days in EMA
    k=2÷(N+1)
    """
    def __init__(self,n):
        self.n = n
        self.k = 2/(n+1)
        self.emaValue = None


    def initialize(self,values):
        if len(values) < self.n:
            raise Exception(f"values length is minor then n ({len(values)} < {self.n})")

        values = values[-self.n:]
        self.emaValue = values[0]
        for v in values:
            self.calculate(v)

    def calculate(self,v):
        self.emaValue = v * self.k + self.emaValue * (1-self.k)
        return self.emaValue


class Macd:
    """
    MACD=12-Period EMA − 26-Period EMA
    SIGNAL=9-Period EMA (macd)
    """
    def __init__(self):
        self.ema26 = Ema(26)
        self.ema12 = Ema(12)
        self.emaSignal = Ema(9)
        self.macd = None
        self.signal = None

    def initialize(self,values):
        if len(values) < self.ema26.n + self.emaSignal.n:
            raise Exception(f"values length is minor then n ({len(values)} < {self.n})")

        values = values[-(self.ema26.n + self.emaSignal.n):]
        self.ema26.initialize(values[0:26])
        self.ema12.initialize(values[0:26])

        macd = []
        for v in values[26:]:
            macd.append(self.ema12.calculate(v) - self.ema26.calculate(v))

        self.emaSignal.initialize(macd)
        self.macd = self.ema12.emaValue - self.ema26.emaValue
        self.signal = self.emaSignal.emaValue

    def calculate(self,v):
        self.macd = self.ema12.calculate(v) - self.ema26.calculate(v)
        self.signal = self.emaSignal.calculate(self.macd)
        return self.macd, self.signal