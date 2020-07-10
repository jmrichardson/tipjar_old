import pickle
import numpy as np
import pandas as pd
import mlfinlab as ml
from pyod.models.knn import KNN
from mlfinlab.features.fracdiff import frac_diff_ffd
import pandas_ta as ta


class Process:

    # def rolling_prediction(self, rolling_window=500):
        # Way too slow!!! TODO: try something a bit more efficient
        # # Don't know of an efficient way to do rolling window on all df and return df, so just iterate
        # data = []
        # for i, row in enumerate(self.pdf.itertuples()):
            # if i+1 < rolling_window: continue
            # print(i)
            # df = self.pdf.iloc[i+1-rolling_window:i, ]
            # date = df.index[-1]
            # prophet = Prophet()
            # prophet.fit(df)
            # mdf = prophet.make_future_dataframe(periods=30, freq='min', include_history=False)
            # mdf.set_index('ds', drop=False, inplace=True)
            # mdf = prophet.predict(mdf)
            # hdf = prophet.make_future_dataframe(periods=5, freq='H', include_history=False)
            # hdf.set_index('ds', drop=False, inplace=True)
            # hdf = prophet.predict(hdf)
            # data.append(pd.Series(date).append(mdf.yhat.iloc[[0, 1, 2, 5, 10, 20, 29]]).append(hdf.yhat).reset_index(drop=True).to_list())
        # self.phat = pd.DataFrame(data).set_index(0, drop=True)

    def indicators(self):
        # TODO: Create feature selection to reduce

        # Momentum
        self.X.ta.ao(append=True)
        self.X.ta.apo(append=True)
        self.X.ta.bop(append=True)
        self.X.ta.cci(append=True)
        self.X.ta.cg(append=True)
        self.X.ta.cmo(append=True)
        self.X.ta.coppock(append=True)
        self.X.ta.fisher(append=True)
        self.X.ta.kst(append=True)
        self.X.ta.macd(append=True)
        self.X.ta.mom(append=True)
        self.X.ta.ppo(append=True)
        self.X.ta.roc(append=True)
        self.X.ta.rsi(append=True)
        self.X.ta.rvi(append=True)
        self.X.ta.stoch(append=True)
        self.X.ta.trix(append=True)
        self.X.ta.tsi(append=True)
        self.X.ta.uo(append=True)
        self.X.ta.willr(append=True)

        # Overlap
        self.X.ta.dema(append=True)
        self.X.ta.ema(append=True)
        self.X.ta.fwma(append=True)
        self.X.ta.hl2(append=True)
        self.X.ta.hlc3(append=True)
        self.X.ta.hma(append=True)
        self.X.ta.kama(append=True)
        self.X.ta.ichimoku(append=True)
        self.X.ta.linreg(append=True)
        self.X.ta.midpoint(append=True)
        self.X.ta.midprice(append=True)
        self.X.ta.ohlc4(append=True)
        self.X.ta.pwma(append=True)
        self.X.ta.rma(append=True)
        self.X.ta.sma(append=True)
        self.X.ta.sinwma(append=True)
        self.X.ta.swma(append=True)
        self.X.ta.t3(append=True)
        self.X.ta.tema(append=True)
        self.X.ta.trima(append=True)
        self.X.ta.vwap(append=True)
        self.X.ta.vwma(append=True)
        self.X.ta.wma(append=True)
        self.X.ta.zlma(append=True)

        # Trend
        self.X.ta.adx(append=True)
        self.X.ta.amat(append=True)
        self.X.ta.aroon(append=True)
        self.X.ta.decreasing(append=True)
        self.X.ta.dpo(append=True)
        self.X.ta.increasing(append=True)
        self.X.ta.linear_decay(append=True)
        self.X.ta.long_run(append=True)
        self.X.ta.qstick(append=True)
        self.X.ta.short_run(append=True)
        self.X.ta.vortex(append=True)

        # Volatility
        self.X.ta.accbands(append=True)
        self.X.ta.atr(append=True)
        self.X.ta.bbands(append=True)
        self.X.ta.donchian(append=True)
        self.X.ta.massi(append=True)
        self.X.ta.natr(append=True)
        self.X.ta.true_range(append=True)

        # Volume
        self.X.ta.ad(append=True)
        self.X.ta.adosc(append=True)
        self.X.ta.aobv(append=True)
        self.X.ta.cmf(append=True)
        self.X.ta.efi(append=True)
        self.X.ta.eom(append=True)
        self.X.ta.mfi(append=True)
        self.X.ta.nvi(append=True)
        self.X.ta.obv(append=True)
        self.X.ta.pvi(append=True)
        self.X.ta.pvol(append=True)
        self.X.ta.pvt(append=True)
        self.X.ta.vp(append=True)
        # help(ta.fisher)

    def save(self, file_name):
        pickle.dump(self, file=open(f"cache/{file_name}.pkl", "wb"))

    def load(file_name):
        return pickle.load(open(f"cache/{file_name}.pkl", "rb"))

    def frac_diff(self, cols=['open', 'high', 'low', 'close'], diff_amt=.4):
        self.X[cols] = frac_diff_ffd(self.X[cols], diff_amt)
        self.X = self.X[self.X['close'].notna()]

    def remove_outliers(self, c=.0001):
        clf = KNN(contamination=c)
        clf.fit(self.X[['open', 'high', 'low', 'close']])
        y = clf.labels_
        self.outliers = self.X[y == 1]
        self.X = self.X[y == 0]


    def dollar_bars(self, frac=1/50, batch_size=1000000, verbose=False):
        # Set threshold to fraction of average daily dollar value (default 1/50)
        d = self.X.close * self.X.volume
        self.threshold = d.resample('D').sum().mean()*frac

        # Sample by dollar value
        df = self.X.reset_index()
        df = df.rename(columns={'date': 'date_time'})
        df = df[['date_time', 'close', 'volume']]
        self.X = ml.data_structures.standard_data_structures.get_dollar_bars(df, threshold=self.threshold,
            batch_size=batch_size, verbose=verbose)
        self.X.set_index('date_time', inplace=True)

    def triple_barrier_label(self, volatility_lookback=50, volatility_scaler=1, triplebar_num_days=3,
        triplebar_pt_sl=[1, 1], triplebar_min_ret=0.003, num_threads=1):

        # extract close series
        close = self.X['close']

        # Compute volatility
        daily_vol = ml.util.get_daily_vol(close, lookback=volatility_lookback)

        # Apply Symmetric CUSUM Filter and get timestamps for events
        cusum_events = ml.filters.cusum_filter(close, threshold=daily_vol.mean() * volatility_scaler)

        # Compute vertical barrier
        vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events, close=close,
            num_days=triplebar_num_days)

        # tripple barier events
        triple_barrier_events = ml.labeling.get_events( close=close, t_events=cusum_events, pt_sl=triplebar_pt_sl,
            target=daily_vol, min_ret=triplebar_min_ret, num_threads=num_threads,
            vertical_barrier_times=vertical_barriers)

        # labels
        labels = ml.labeling.get_bins(triple_barrier_events, close)
        labels = ml.labeling.drop_labels(labels)

        # merge labels and triple barrier events
        triple_barrier_info = pd.concat([triple_barrier_events.t1, labels], axis=1)
        triple_barrier_info.dropna(inplace=True)

        self.Xtbl = self.X.reindex(triple_barrier_info.index)
        self.ytbl = triple_barrier_info.bin.astype(int)


    def tbl_sliding_window(self, sw=120, sktime=True, mcfly=True):
        sw = 120
        case = 0

        Xsk=pd.DataFrame(columns=self.Xtbl.columns)
        Xmf=[]

        for i in self.Xtbl.index:
            start = self.X.index.get_loc(i)-sw
            end = self.X.index.get_loc(i)
            window = self.X.iloc[start:end, ]
            window_np = window.to_numpy()

            if mcfly: Xmf.append(window_np)

            if sktime:
                Xsk = Xsk.append(pd.Series([np.nan]), ignore_index=True)
                for name, col in window.iteritems():
                    Xsk.at[case, name] = col
                case = case + 1

        # Final processing
        if mcfly: Xmf = np.stack(Xmf)
        if sktime: Xsk.drop([0], axis=1, inplace=True)





