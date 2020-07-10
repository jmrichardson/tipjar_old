import backtrader as bt
from tipjar.backtest.strategy import Strategy


class PandasDataExt(bt.feeds.PandasData):
    lines = ('yhat',)
    params = (('yhat', 5), )


class BackTest:


    def backtest(self):

        self.cerebro = bt.Cerebro()
        self.cerebro.addstrategy(Strategy)
        data = PandasDataExt(dataname=self.btdf)
        self.cerebro.adddata(data)

        self.cerebro.broker.setcash(100000.0)
        self.cerebro.broker.setcommission(commission=0.001)
        print(f'Starting Portfolio Value: {self.cerebro.broker.getvalue()}')

        self.cerebro_run = self.cerebro.run()
        print(f'Final Portfolio Value: {self.cerebro.broker.getvalue()}')

