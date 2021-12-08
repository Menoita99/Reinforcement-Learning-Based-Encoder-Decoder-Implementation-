import configparser
from mysql import connector
import pandas as pd
import datetime

from src.indicator import Ema, Macd


class DataProcessor():

    def __init__(self):
        self._loadData()


    def _loadData(self):
        config = configparser.RawConfigParser()
        config.read('application.properties')
        mydb = connector.connect(
            host=config.get('Database', 'database.host'),
            user=config.get('Database', 'database.user'),
            password=config.get('Database', 'database.password'),
            database=config.get('Database', 'database.db')
        )

        mycursor = mydb.cursor()

        self.timeFrames = [('M15',15*60),('M30',30*60),('H1',60*60),('H4',4*60*60),('D',24*60*60)]
        self.markets = ['EUR_USD']
        self.lookupTable = []
        self.data = []
        # TODO for market
        # for market in self.markets:
        select = "SELECT open, high, low, close, volume, datetime " \
                 "FROM forexdb.candlestick " \
                 f"where market='{self.markets[0]}' and timeframe='{self.timeFrames[0][0]}' and datetime > '2010-01-01 00:00:00' " \
                 "order by datetime asc;"

        mycursor.execute(select)
        mainTimeFrameValues = mycursor.fetchall()

        for i,mainTimeFrameValue in enumerate(mainTimeFrameValues):
            self.data.append(mainTimeFrameValue[:-1])
            self.lookupTable.append([i])


        for i in range(1,len(self.timeFrames)):
            if i == 0:
                continue
            timeframe = self.timeFrames[i]
            lastCandleFetched = None
            print(f"fetching timeframe: {timeframe}")

            for idx, mainTimeFrameValue in enumerate(mainTimeFrameValues):
                datetimeToFetch = self.convertDateTime(mainTimeFrameValue[5],timeframe[1])
                print(f"{mainTimeFrameValue[5]} {datetimeToFetch}")
                if i > 2 and datetimeToFetch.hour % 2 != 0:
                    datetimeToFetch = datetimeToFetch - datetime.timedelta(seconds=60*60)

                if lastCandleFetched is None or lastCandleFetched[5] != datetimeToFetch:
                    candle = self.getCandle(datetimeToFetch, mycursor, timeframe)
                    self.data.append(candle[0][:-1])
                    lastCandleFetched = candle[0]
                self.lookupTable[idx].append(len(self.data)-1)

        print("Load complete")
        self.processDataSimple()
        self.writeState()


    # (atual / prev) - 1
    # open, high, low, close, volume
    """
    percetage relative to it's relative ( open / open , close /close ... )
      %  open  |  high  |  low  |  close  | ema 200  | ema 50 | macd | macd_signal
    15M
    30M
    1H
    4H
    D
    """
    def processDataToPercentageWithVolume(self):
        self.states = []

        for i in range(len(self.lookupTable)):
            currentCandles = self.lookupTable[i]
            prevCandles = self.lookupTable[max(i-1,0)]

            state = []
            for j in range(len(currentCandles)):
                newCandle = []
                if currentCandles[j] != prevCandles[j]:
                    for k in range(len(currentCandles[j])):
                        newCandle.append(currentCandles[j][k] / prevCandles[j][k] - 1)
                else:
                    open = currentCandles[j][0] / prevCandles[j][0] - 1
                    #TODO
                    high = None
                    low = None
                    close = None
                    Volume = None

                state.append(newCandle)
            self.states.append(state)

    """
        close  | ema 200  | ema 50 | macd | macd_signal
    15M
    """
    def processDataSimple(self):
        self.priceData = []
        self.states = []
        self.initIndicators(self.getInitIndicatorsArray())
        for entry in self.lookupTable[self.startIdx:]:
            close = self.data[entry[0]][3]
            self.priceData.append(close)
            macd15M_macd = self.macd15M.calculate(close)[0]
            macd15M_signal = self.macd15M.calculate(close)[1]
            self.states.append([close, self.ema15M_200.calculate(close),self.ema15M_50.calculate(close),self.ema15M_13.calculate(close),self.ema15M_7.calculate(close),macd15M_macd,macd15M_signal])


    def getInitIndicatorsArray(self):
        initInd = [[] for _ in self.lookupTable[0]]
        self.startIdx = None
        for i in range(len(self.lookupTable)):
            currentCandles = self.lookupTable[i]
            prevCandles = self.lookupTable[max(i - 1, 0)]

            for j in range(len(currentCandles)):
                if currentCandles[j] != prevCandles[j]:
                    initInd[j].append(self.data[currentCandles[j]][3])

            for array in initInd:
                if len(array) > 200:
                    array.pop(0)

            if len(initInd[-1]) >= 200:
                self.startIdx = i + 1
                break
        return initInd




    def getCandle(self, datetimeToFetch, mycursor, timeframe):
        select = "SELECT open, high, low, close, volume, datetime " \
                 "FROM forexdb.candlestick " \
                 f"where market='{self.markets[0]}' and timeframe='{timeframe[0]}' " \
                 f"and datetime >= '{datetimeToFetch}' " \
                 f"and datetime < '{datetimeToFetch + datetime.timedelta(seconds=timeframe[1])}' " \
                 "order by datetime asc;"
        mycursor.execute(select)
        candle = mycursor.fetchall()
        if len(candle) == 0:
            select = "SELECT open, high, low, close, volume, datetime " \
                     "FROM forexdb.candlestick " \
                     f"where market='{self.markets[0]}' and timeframe='{self.timeFrames[0][0]}' " \
                     f"and datetime >= '{datetimeToFetch}' " \
                     f"and datetime < '{datetimeToFetch + datetime.timedelta(seconds=timeframe[1])}' " \
                     "order by datetime asc;"
            mycursor.execute(select)
            print(select)
            candles = mycursor.fetchall()
            if len(candles) == 0:
                select = "SELECT open, high, low, close, volume, datetime " \
                         "FROM forexdb.candlestick " \
                         f"where market='{self.markets[0]}' and timeframe='{self.timeFrames[0][0]}' " \
                         f"and datetime >= '{datetimeToFetch + datetime.timedelta(seconds=timeframe[1])}' " \
                         f"and datetime < '{datetimeToFetch + datetime.timedelta(seconds=timeframe[1]) + datetime.timedelta(seconds=timeframe[1])}' " \
                         "order by datetime asc;"
                mycursor.execute(select)
                print(select)
                candles = mycursor.fetchall()
            candle = self.generateCandle(candles)
        elif len(candle) > 1:
            raise Exception(f"MYSQL RETURNED {candle} , IT SHOULD HAVE ONLY 1 ELEM\n{select}")
        return candle



    def generateCandle(self, candles):
        if len(candles) == 0:
            raise Exception("Could not generate Candle from lowest timeframe")
        open = candles[0][0]
        close = candles[-1][3]
        high = 0
        low = float('inf')
        volume = 0
        for candle in candles:
            if candle[1] > high:
                high = candle[1]
            if candle[2] < low:
                low = candle[2]
            volume += candle[4]
        return [(open, high, low, close, volume, candles[0][5])]



    def convertDateTime(self,dateTime,timeframe_in_seconds):
        return dateTime - datetime.timedelta(seconds=(dateTime.timestamp() % timeframe_in_seconds))



    def initIndicators(self,initArray):
        self.ema15M_200 = Ema(200)
        self.ema15M_50 = Ema(50)
        self.ema15M_13 = Ema(13)
        self.ema15M_7 = Ema(7)
        self.macd15M = Macd()

        self.ema30M_200 = Ema(200)
        self.ema30M_50 = Ema(50)
        self.macd30M = Macd()

        self.ema1H_200 = Ema(200)
        self.ema1H_50 = Ema(50)
        self.macd1H = Macd()

        self.ema4H_200 = Ema(200)
        self.ema4H_50 = Ema(50)
        self.macd4H = Macd()

        self.emaD_200 = Ema(200)
        self.emaD_50 = Ema(50)
        self.macdD = Macd()

        self.ema15M_200.initialize(initArray[0])
        self.ema15M_50.initialize(initArray[0])
        self.ema15M_13.initialize(initArray[0])
        self.ema15M_7.initialize(initArray[0])
        self.macd15M.initialize(initArray[0])

        self.ema30M_200.initialize(initArray[1])
        self.ema30M_50.initialize(initArray[1])
        self.macd30M.initialize(initArray[1])

        self.ema1H_200.initialize(initArray[2])
        self.ema1H_50.initialize(initArray[2])
        self.macd1H.initialize(initArray[2])

        self.ema4H_200.initialize(initArray[3])
        self.ema4H_50.initialize(initArray[3])
        self.macd4H.initialize(initArray[3])

        self.emaD_200.initialize(initArray[4])
        self.emaD_50.initialize(initArray[4])
        self.macdD.initialize(initArray[4])




    def generateState(self):
        prevCandlesIdx = self.lookupTable[max(self.currentCandle-1,0)]
        candlesIdx = self.lookupTable[self.currentCandle]

        self.currentPrice = round((self.data[candlesIdx[0]][3] / 100 + 1) * self.currentPrice, 5)

        candle15M, candle30M, candle1H, candle4H, candleD = self.processCandles(prevCandlesIdx,candlesIdx)

        if self.data[candlesIdx[0]] != self.data[prevCandlesIdx[0]]:
            self.ema15M_200.calculate(candle15M[3])
            self.ema15M_50.calculate(candle15M[3])
            self.macd15M.calculate(candle15M[3])

        if self.data[candlesIdx[1]] != self.data[prevCandlesIdx[1]]:
            self.ema30M_200.calculate(candle30M[3])
            self.ema30M_50.calculate(candle30M[3])
            self.macd30M.calculate(candle30M[3])

        if self.data[candlesIdx[2]] != self.data[prevCandlesIdx[2]]:
            self.ema1H_200.calculate(candle1H[3])
            self.ema1H_50.calculate(candle1H[3])
            self.macd1H.calculate(candle1H[3])

        if self.data[candlesIdx[3]] != self.data[prevCandlesIdx[3]]:
            self.ema4H_200.calculate(candle4H[3])
            self.ema4H_50.calculate(candle4H[3])
            self.macd4H.calculate(candle4H[3])

        if self.data[candlesIdx[4]] != self.data[prevCandlesIdx[4]]:
            self.emaD_200.calculate(candleD[3])
            self.emaD_50.calculate(candleD[3])
            self.macdD.calculate(candleD[3])

        candle15M.append(self.ema15M_50.emaValue)
        candle15M.append(self.ema15M_200.emaValue)
        candle15M.append(self.macd15M.macd)
        candle15M.append(self.macd15M.signal)

        candle30M.append(self.ema30M_50.emaValue)
        candle30M.append(self.ema30M_200.emaValue)
        candle30M.append(self.macd30M.macd)
        candle30M.append(self.macd30M.signal)

        candle1H.append(self.ema1H_50.emaValue)
        candle1H.append(self.ema1H_200.emaValue)
        candle1H.append(self.macd1H.macd)
        candle1H.append(self.macd1H.signal)

        candle4H.append(self.ema4H_50.emaValue)
        candle4H.append(self.ema4H_200.emaValue)
        candle4H.append(self.macd4H.macd)
        candle4H.append(self.macd4H.signal)

        candleD.append(self.emaD_50.emaValue)
        candleD.append(self.emaD_200.emaValue)
        candleD.append(self.macdD.macd)
        candleD.append(self.macdD.signal)

        self.addPositionToCandle(candle15M)
        self.addPositionToCandle(candle30M)
        self.addPositionToCandle(candle1H)
        self.addPositionToCandle(candle4H)
        self.addPositionToCandle(candleD)

        return[candle15M,candle30M,candle1H,candle4H,candleD]




    def writeState(self):
        with pd.option_context('display.precision', 10):
            df = pd.DataFrame(self.states, columns=['close', 'ema200', 'ema50', 'ema13', 'ema7', 'macd_macd', 'macd_signal'])
            df.to_csv('C:\\Users\\rui.menoita\\Documents\\GitHub\\Reinforcement-Learning-Based-Encoder-Decoder-Implementation-\\src\\data\\close ema200 ema50 ema13 ema7 macd_macd macd_signal.csv'
                      ,doublequote=True,index=False,sep=';')





DataProcessor()
