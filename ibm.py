
from tipjar.kibot import Kibot
# ibm = Kibot(f'data/kibot/IBM/minute/IBM.txt', nrows=9999999999)
ibm = Kibot(f'data/kibot/IBM/minute/IBM.txt', nrows=999999999)

# Remove outliers
ibm.remove_outliers()

ibm.save("IBM_load")
ibm = Kibot.load("IBM_load")

# Add features
ibm.matrix_profile()
ibm.add_time()

# Label
ibm.triple_barrier_label()


# Make stationary
ibm.frac_diff()

# Segment into time series sequences
ibm.segment(sliding_window=1440)
ibm.save("IBM_segment")

# I don't think we need to scale because did differencing
# ibm.segment_scale()
ibm.train_val_test_split()
ibm.save("IBM_split")

# ibm = Kibot.load("IBM_tbl")
# self=ibm


ibm.mcfly_model()
ibm.save("mcfly")












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


