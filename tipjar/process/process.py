import pickle
import numpy as np
import pandas as pd
import mlfinlab as ml
from pyod.models.knn import KNN
from mlfinlab.features.fracdiff import frac_diff_ffd
from tslearn.utils import to_pyts_dataset
from tslearn.utils import to_stumpy_dataset
from tslearn.utils import to_sktime_dataset
from tslearn.utils import to_cesium_dataset
from tslearn.utils import to_tsfresh_dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas_ta as ta
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matrixprofile as mp
from sktime.transformers.series_as_features.rocket import Rocket
from deltapy import transform, interact, mapper, extract


class Process:

    def x_aug(self):
        # # Transformations
        # df_out = transform.triple_exponential_smoothing(df_out.copy(), ["close"], 12, .2, .2, .2, 0);
        # df_out = transform.harmonicradar_cw(df_out.copy(), ["close"], 0.3, 0.2);
        # df_out = transform.butter_lowpass_filter(df_out.copy(), ["close"], 4)
        # df_out = transform.instantaneous_phases(df_out.copy(), ["close"])
        # df_out = transform.saw(df_out.copy(), ["close", "open"]);
        # # df_out = interact.lowess(df_out.copy(), ["open","volume"], self.X["close"], f=0.25, iter=3);
        # df_out = transform.instantaneous_phases(df_out.copy(), ["close"])
        # df_out = transform.perd_feat(df_out.copy(), ["close"]);
        # df_out = transform.harmonicradar_cw(df_out.copy(), ["close"], 0.3, 0.2);
        # df_out = transform.multiple_lags(df_out.copy(), start=1, end=3, columns=["close"])
        # # df = transform.kalman_feat(self.X.copy(), ["close"])
        # # df = transform.bkb(self.X.copy(), ["close"])
        # # df_out = transform.operations(self.X.copy(),["close"])
        # # df_out = transform.naive_dec(self.X.copy(), ["close","open"])
        # # df_out = transform.perd_feat(self.X.copy(),["close"])
        # # df_out = transform.fft_feat(self.X.copy(), ["close"])
        # # df_out = transform.modify(self.X.copy(),["close"]);
        # # df_out = transform.multiple_rolling(self.X.copy(), columns=["close"]);
        # # df_out = interact.autoregression(self.X.copy())
        # # df_out = interact.muldiv(self.X.copy(), ["close","open"])
        # # df_out = interact.decision_tree_disc(self.X.copy(), ["close"])
        # # df_out = interact.quantile_normalize(self.X.copy(), drop=["close"]);
        # # df_out['new'] = mapper.pca_feature(df.copy(), variance_or_components=0.9, n_components=8,non_linear=False)
        # # df_out = mapper.cross_lag(df.copy())
        # # df_out = mapper.encoder_dataset(df.copy(), dimesions=4);
        # # df_out = interact.genetic_feat(self.X.copy())

        # # Interactions
        pass

    def seg_np(self):
        for i, df in enumerate(self.Xseg):
            self.Xseg[i] = self.Xseg[i].to_numpy()
        self.Xseg = np.stack(self.Xseg)

    def seg_matrix_profile(self, windows=[5, 10]):
        for i, df in enumerate(self.Xseg):
            self.mp = mp.compute(np.array(self.Xseg[i].close), windows=windows)
            for k, window in enumerate(windows):
                self.Xseg[i]['mp' + str(window)] = pd.Series(self.mp['pmp'][k]).values
            self.Xseg[i] = self.Xseg[i].replace([np.inf, -np.inf], np.nan)

    def rocket(self):
        self.Xskt = to_sktime_dataset(self.Xseg['close'])
        rocket = Rocket()
        X = rocket.fit(self.Xskt)
        X_train_transform = rocket.transform(self.Xskt)
        from sklearn.linear_model import RidgeClassifierCV

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(X_train_transform, self.yseg)

    def add_time(self):
        """Adds time features to X"""
        self.X['minute'] = self.X.index.minute
        self.X['hour'] = self.X.index.hour
        self.X['dayofweek'] = self.X.index.dayofweek

    def indicators(self):
        """Adds technical analysis indicator features to X"""

        # Add new bulk strategy method to add indicators!!!

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

        self.X.dropna(how='all', axis=1, inplace=True)

    def seg_rocket(self):
        """Rocket"""

        # Temporarily suppress settingwithcopywarning
        from pandas import options
        options.mode.chained_assignment = None

        for i, df in enumerate(self.Xseg):
            print(i)
            #  Trend
            self.Xseg[i].ta.adx(append=True)

        options.mode.chained_assignment = "warn"

    def save(self, file_name):
        """Save equity object"""
        pickle.dump(self, file=open(f"tmp/{file_name}.pkl", "wb"))

    def load(file_name):
        """Load equity object"""
        return pickle.load(open(f"tmp/{file_name}.pkl", "rb"))

    def frac_diff(self, cols=['open', 'high', 'low', 'close'], diff_amt=.4):
        """Detrend (make stationary)"""
        self.X[cols] = frac_diff_ffd(self.X[cols], diff_amt)
        self.X = self.X[self.X['close'].notna()]

    def remove_outliers(self, c=.0001):
        """Remove outliers
        c: Estimated percent of outliers"""
        clf = KNN(contamination=c)
        clf.fit(self.X[['open', 'high', 'low', 'close']])
        y = clf.labels_
        self.outliers = self.X[y == 1]
        self.X = self.X[y == 0]


    def dollar_bars(self, frac=1/50, batch_size=1000000, verbose=False):
        """If using tick data, sample by dollar"""
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

    def triple_barrier_label(self, lookback=30, scaler=1, num_hours=1, pt_sl=[1, 1], min_ret=0.003, num_threads=1):
        """Label samples using triple barrier
        lookback: Number of previous days to calculate volatility
        num_hours: Number of hours in future to place vertical bar
        min_ret: Required return for long/short
        Returns: self.Xtbl
        Note:  This must be done first then sync'd later
        """

        # Compute volatility
        # vol = ml.util.get_yang_zhang_vol(self.X.open, self.X.high, self.X.low, self.X.close, window=lookback)
        # vol.dropna(inplace=True)
        vol = ml.util.get_daily_vol(self.X.close, lookback=lookback)

        # Apply Symmetric CUSUM Filter and get timestamps for events
        cusum_events = ml.filters.cusum_filter(self.X.close, threshold=vol.mean() * scaler)

        # Restrict labels between time range
        cusum_events = cusum_events.to_frame().between_time('09:30', '16:00').index

        # Compute vertical barrier
        vertical_barriers = ml.labeling.add_vertical_barrier(t_events=cusum_events, close=self.X.close, num_hours=num_hours)

        # tripple barier events
        triple_barrier_events = ml.labeling.get_events(close=self.X.close, t_events=cusum_events, pt_sl=pt_sl,
            target=vol, min_ret=min_ret, num_threads=num_threads, vertical_barrier_times=vertical_barriers)

        # labels
        labels = ml.labeling.get_bins(triple_barrier_events, self.X.close)
        labels = ml.labeling.drop_labels(labels)

        # merge labels and triple barrier events
        triple_barrier_info = pd.concat([triple_barrier_events.t1, labels], axis=1)
        triple_barrier_info.dropna(inplace=True)

        self.Xtbl = self.X.reindex(triple_barrier_info.index)
        self.ytbl = triple_barrier_info.bin.astype(int)

        # Average number of long (pt)
        self.daily_long_avg = self.ytbl[self.ytbl == 1].resample('B').count().mean()
        self.daily_long_avg = round(self.daily_long_avg, 2)

    # def xtbl_sync(self):
        # """Reindex Xtbl after X OHLC modifications such as frac_diff"""
        # self.Xtbl = self.X.reindex(self.Xtbl.index)
        # self.Xtbl.dropna(inplace=True)

    def seg_df(self, sliding_window=120):
        '''Create list of dataframes'''
        self.Xseg = []
        self.yseg = []
        for i in self.Xtbl.index:
            if i in self.X.index:
                start = self.X.index.get_loc(i)-sliding_window
                end = self.X.index.get_loc(i)
                # window = self.X.iloc[start:end, ].to_numpy()
                window = self.X.iloc[start:end, ]
                if len(window) == sliding_window:
                    self.Xseg.append(window)
                    self.yseg.append(self.ytbl.loc[i])
        # self.Xseg = np.stack(self.Xseg)
        # bin into columns: col0=1, col1=-1, col2=0
        # # bin = LabelBinarizer().fit_transform(self.yseg)
        # enc = OneHotEncoder()
        # enc = enc.fit(bin)
        # self.yseg = enc.transform(bin).toarray()
        self.yseg = LabelBinarizer().fit_transform(self.yseg)

    def train_val_test_split(self, test=.2, val=.3):
        self.Xseg_train, self.Xseg_test, self.yseg_train, self.yseg_test = train_test_split(self.Xseg, self.yseg,
            test_size=test, random_state=0)
        self.Xseg_train, self.Xseg_val, self.yseg_train, self.yseg_val = train_test_split(self.Xseg_train, self.yseg_train,
            test_size=val, random_state=0)

    def tslearn(self):
        pass

    def seglearn(self):
        pass

    def mcfly(self):
        pass

    def pyts(self):
        self.Xseg = to_pyts_dataset(self.Xseg)

    def stumpy(self):
        self.Xseg = to_stumpy_dataset(self.Xseg)

    def sktime(self):
        self.Xskt = to_sktime_dataset(self.Xseg)

    def cesium(self):
        self.Xseg = to_cesium_dataset(self.Xseg)

    def tsfresh(self):
        self.Xseg = to_tsfresh_dataset(self.Xseg)


    def segment_scale(self):
        self.Xseg = TimeSeriesScalerMeanVariance(mu=0., std=1).fit_transform(self.Xseg)

