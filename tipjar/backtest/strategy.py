from io import BytesIO
from csv import writer
import backtrader as bt


class Strategy(bt.Strategy):

    def __init__(self):
        self.close = self.datas[0].close
        self.yhat = self.datas[0].yhat
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.io = BytesIO

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f} Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f} Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        self.log(f'Close, {self.close[0]:.2f}, line {len(self)}')

        if self.order:
            raise("This shouldn't happen")


        if not self.position:
            if self.yhat[0] == 1:
                self.log(f'BUY CREATE, {self.close[0]}')
                self.order = self.buy()
        else:
            self.log(f'SELL CREATE, {self.close[0]}')
            self.order = self.sell()


