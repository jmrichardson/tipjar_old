
from tipjar.kibot import Kibot
# ibm = Kibot(f'data/kibot/IBM/minute/IBM.txt', nrows=9999999999)
ibm = Kibot(f'data/kibot/IBM/minute/IBM.txt', nrows=9999)

# Remove outliers
ibm.remove_outliers()

# Label
ibm.triple_barrier_label()

# Add features
ibm.add_time()

# Make stationary
ibm.frac_diff()

# Segment into time series
ibm.segment(sliding_window=300)
ibm.save("IBM_segment")

# I don't think we need to scale because did differencing
# ibm.segment_scale()
ibm.train_val_test_split()
ibm.save("IBM_split")

# ibm = Kibot.load("IBM_tbl")
# self=ibm


ibm.mcfly_model()




from numpy import array
from numpy.random import rand
Xt = array([rand(100,5), rand(200,5), rand(50,5)])
Xc = rand(3,2)
from seglearn.base import TS_Data
import pandas as pd
df = pd.DataFrame(Xc)
df['ts_data'] = Xt
X = TS_Data.from_df(df)
y = [0, 1, 1]



from seglearn.transform import Segment
from tslearn.utils import to_time_series_dataset
from tslearn.utils import to_seglearn_dataset
self.Xts = to_time_series_dataset([self.X.open.values.tolist(), self.X.high.values.tolist(), self.X.close.values.tolist()])
self.yts = self.ytbl.values.tolist()
seg = Segment(width=5)
seg.fit_transform(self.Xts, self.yts)




ibm.train_test_split_X()
self = ibm
# ibm.prophet(rolling_window=675)


# ibm.indicators()
ibm.save("IBM_indicators")
ibm.frac_diff(['open', 'high', 'low', 'close'], .4)
ibm.triple_barrier_label()
ibm.save("IBM_tbl")

from freetrade.kibot import Kibot
if 'ibm' not in locals(): ibm = Kibot.load("IBM")
ibm.train_test_split(test_size=.02)
ibm.dummy_model("prior")
ibm.evaluate()
print(ibm.mean_accuracy)
ibm.save("IBM_model")

from freetrade.kibot import Kibot
if 'ibm' not in locals(): ibm = Kibot.load("IBM_model")
ibm.backtest()


# ibm.model.predict(ibm.X_test)
# ibm.cerebro.plot()
# Take a look at finta on github


