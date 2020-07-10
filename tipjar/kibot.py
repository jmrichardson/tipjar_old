import pandas as pd
from tipjar.process.process import Process
from tipjar.model.model import Model
from tipjar.backtest.backtrader import BackTest


class Kibot(Process, Model, BackTest):

    def __init__(self, path, nrows=None):
        # X must have format of open, high, low, close volume - with datetime index
        self.path = path
        self.raw = pd.read_csv(path, names=['date', 'time', 'open', 'high', 'low', 'close', 'volume'], nrows=nrows)
        self.raw['date'] = self.raw.date + ' ' + self.raw.time
        self.raw.drop(columns=['time'], inplace=True)
        self.raw['date'] = pd.to_datetime(self.raw['date'])
        self.raw.set_index('date', inplace=True)

        # TODO: Not sure i should do this yet
        # I think an hour feature will handle off hours
        self.raw = self.raw.between_time('09:30', '14:00')
        self.X = self.raw.copy()

