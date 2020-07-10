import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn import metrics
# from fbprophet import Prophet
# from fbprophet.diagnostics import cross_validation, performance_metrics
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import os
import mcfly
import tensorflow as tf



class Model:

    def dummy_model(self, strategy="stratified"):
        self.model = DummyClassifier(strategy=strategy, random_state=42)
        self.model.fit(self.X_train, self.y_train.values.ravel())

    def random_forest_model(self):
        self.model = RandomForestClassifier(n_jobs=-1, n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train.values.ravel())

    # This doesn't work: TODO: fix predict
    def segment_KNN_model(self):
        self.model = KNeighborsTimeSeriesClassifier(n_neighbors=1)
        self.model.fit(self.Xs, self.ys)
        # print(self.model.predict(self.Xs[0]))

    def mcfly_model(self):
        # Cross validation takes too long with DL

        self.num_models = 4
        self.num_classes = self.yseg_train.shape[1]
        self.metric = 'accuracy'
        self.models = mcfly.modelgen.generate_models(self.Xseg_train.shape,
                                            number_of_classes=self.num_classes,
                                            number_of_models=self.num_models,
                                            metrics=[self.metric])

        from mcfly.find_architecture import train_models_on_samples
        resultpath = os.path.join('.', 'models')
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        outputfile = os.path.join(resultpath, 'modelcomparison.json')
        self.histories, self.val_accuracies, self.val_losses = train_models_on_samples(self.Xseg_train, self.yseg_train,
                                                                        self.Xseg_val, self.yseg_val,
                                                                        self.models, nr_epochs=20,
                                                                        subset_size=300,
                                                                        early_stopping_patience=5,
                                                                        verbose=True,
                                                                        outputfile=outputfile,
                                                                        metric=self.metric)
        print('Details of the training process were stored in ', outputfile)

        self.modelcomparisons = pd.DataFrame({'model': [str(params) for model, params, model_types in self.models],
                                         'model-type': [str(model_types) for model, params, model_types in self.models],
                                         'train_{}'.format(self.metric): [history.history[self.metric][-1] for history in
                                                                     self.histories],
                                         'train_loss': [history.history['loss'][-1] for history in self.histories],
                                         'val_{}'.format(self.metric): [history.history['val_{}'.format(self.metric)][-1] for
                                                                   history in self.histories],
                                         'val_loss': [history.history['val_loss'][-1] for history in self.histories]
                                         })
        self.modelcomparisons.to_csv(os.path.join(resultpath, 'modelcomparisons.csv'))
        self.modelcomparisons

    def seglearn_model(self):
        from tensorflow.python.keras.layers import Dense, LSTM, Conv1D
        from tensorflow.python.keras.models import Sequential
        from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import train_test_split
        from seglearn.pipe import Pype

        def crnn_model(width=120, n_vars=5, n_classes=2, conv_kernel_size=5,
                       conv_filters=3, lstm_units=3):
            input_shape = (width, n_vars)
            model = Sequential()
            model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                             padding='valid', activation='relu', input_shape=input_shape))
            model.add(Conv1D(filters=conv_filters, kernel_size=conv_kernel_size,
                             padding='valid', activation='relu'))
            model.add(LSTM(units=lstm_units, dropout=0.1, recurrent_dropout=0.1))
            model.add(Dense(n_classes, activation="softmax"))
            model.compile(loss='categorical_crossentropy', optimizer='adam',
                          metrics=['accuracy'])
            return model

        # X_train, X_test, y_train, y_test = train_test_split(self.Xseg, self.yenc, test_size=0.25, random_state=42)
        pipe = Pype([('crnn', KerasClassifier(build_fn=crnn_model, epochs=1, batch_size=256, verbose=0))])
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)

    def evaluate(self):
        self.yhat = self.model.predict(self.X_test)
        self.precision = metrics.precision_score(self.y_test, self.yhat)
        self.recall = metrics.recall_score(self.y_test, self.yhat)
        self.f1 = metrics.f1_score(self.y_test, self.yhat)
        self.mean_accuracy = self.model.score(self.X_test, self.y_test)
        self.btdf = self.X_test.copy()
        self.btdf['yhat'] = self.yhat

    def train_test_split_X(self, test_size=.3, shuffle=False):
        self.X_train, self.X_test, = train_test_split(self.X, test_size=test_size, random_state=0, shuffle=False)

    def train_test_split_Xy(self, test_size=.3, random_state=0, shuffle=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=0, shuffle=False)


