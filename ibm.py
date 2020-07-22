
from deltapy import transform, interact, mapper, extract
from tsaug import *
from tipjar.kibot import Kibot
# ibm = Kibot(f'data/kibot/IBM/minute/IBM.txt')
ibm = Kibot(f'data/kibot/IBM/minute/IBM.txt', nrows=99999)
self = ibm


# Remove outliers
ibm.remove_outliers()

# Add features which should be applied to ALL of X
# ibm.indicators()

ibm.save("IBM_load")
# ibm = Kibot.load("IBM_load")


# Dont do this yet....
# ibm.matrix_profile()

# ibm.save("IBM_mp")
# ibm = Kibot.load("IBM_mp")

# Add features to all of X
ibm.add_time()
ibm.indicators()

# Label
ibm.triple_barrier_label()

ibm.save("IBM_tbl")
ibm = Kibot.load("IBM_tbl")

### Segment into list of dataframe time sequences Xseg
# 2400 is ~5 8 hour days
# 124800 is ~1 year

ibm.seg_df(sliding_window=124800)

a = ibm.Xseg[0]

# Add matrix profile
ibm.seg_matrix_profile()


### Convert to Xseg 3D numpy
ibm.seg_np()


ibm.rocket()

# ibm.save("IBM_segment")


# Make stationary
# ibm.frac_diff()


import pandas as pd
from darts import TimeSeries
from darts.preprocessing import ScalerWrapper
from darts.models.tcn_model import TCNModel
from darts.models.prophet import Prophet
from darts.models.exponential_smoothing import ExponentialSmoothing
from darts.models.fft import FFT
from darts.models.theta import Theta
ts = TimeSeries.from_dataframe(a[:5000], time_col=None, value_cols=['close'])
ts.describe()
ts.plot()

model = Prophet()
model = ExponentialSmoothing()
model = Theta()
model = FFT()
model.fit(ts)
b = model.predict(2400)
b.plot()


len(ts)

transformer = ScalerWrapper()
train_transformed = transformer.fit_transform(series)


# Create TCNModel instance
my_model = TCNModel(
    n_epochs=200,
    input_length=50,
    output_length=13,
    dropout=0.1,
    dilation_base=2,
    weight_norm=True,
    kernel_size=4,
    num_filters=3
)

my_model.fit(train_transformed, verbose=True)






# I don't think we need to scale because did differencing
# ibm.segment_scale()
ibm.train_val_test_split()
ibm.save("IBM_split")

# ibm = Kibot.load("IBM_tbl")
# self=ibm


ibm.mcfly_model()
ibm.save("mcfly")





import pandas as pd
from darts import TimeSeries







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




import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.datasets import load_arrow_head  # univariate dataset
from sktime.datasets.base import load_japanese_vowels  # multivariate dataset
from sktime.transformers.series_as_features.rocket import Rocket



X_train, y_train = load_arrow_head(split="train", return_X_y=True)

rocket = Rocket() # by default, ROCKET uses 10,000 kernels
rocket.fit(X_train)
X_train_transform = rocket.transform(X_train)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting.forecasting import plot_ys


y = load_airline()
fig, ax = plot_ys(y)
ax.set(xlabel="Time", ylabel="Number of airline passengers");



y_train, y_test = temporal_train_test_split(y, test_size=36)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting.forecasting import plot_ys

%matplotlib inline



y = load_airline()
fig, ax = plot_ys(y)
ax.set(xlabel="Time", ylabel="Number of airline passengers");

y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=36)
print(y_train.shape[0], y_test.shape[0])

from sktime.forecasting.compose import ReducedRegressionForecaster
from sklearn.neighbors import KNeighborsRegressor

regressor = KNeighborsRegressor(n_neighbors=1)
forecaster = ReducedRegressionForecaster(regressor=regressor, window_length=12, strategy="recursive")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh)
plot_ys(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]);
smape_loss(y_test, y_pred)




import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_arrow_head
from sktime.series_as_features.compose import FeatureUnion
from sktime.transformers.series_as_features.compose import RowTransformer
from sktime.transformers.series_as_features.reduce import Tabularizer
from sktime.transformers.series_as_features.segment import \
    RandomIntervalSegmenter
from sktime.utils.time_series import time_series_slope
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf



X, y = load_arrow_head(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)