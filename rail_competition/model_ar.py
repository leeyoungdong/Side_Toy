from tensorflow.keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from eda import *
from model_loss import *


class DeepARModel:
    def __init__(self, N_TIMESTEPS=20, N_FEATURES=36, batch_size=64, loss_fn=myMSETF()):
        self.N_TIMESTEPS = N_TIMESTEPS
        self.N_FEATURES = N_FEATURES
        self.batch_size = batch_size
        self.loss_fn = loss_fn.loss
        self.model = self.build_model()
        self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

    def build_model(self):
        inputs = Input(shape=(self.N_TIMESTEPS, self.N_FEATURES))
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(128, return_sequences=False)(x)
        x = Dropout(0.5)(x)
        x = RepeatVector(self.N_TIMESTEPS)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.5)(x)
        x = TimeDistributed(Dense(2))(x)
        model = Model(inputs=inputs, outputs=x[:, -1, :])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=self.loss_fn, metrics=['mae'])
        return model

    def _prepare_data(self, data_type):
        data_c, data_s, data_c_test, data_s_test = getdata()
        if data_type == 's':
            data, data_test = data_s, data_s_test
        else:
            data, data_test = data_c, data_c_test
        
        X_train_r, y_train_r = prepare_data_right(data)  
        X_test_r, y_test_r = prepare_data_right(data_test)
        
        X_train_l, y_train_l = prepare_data_left(data)  
        X_test_l, y_test_l = prepare_data_left(data_test)

        return X_train_r, y_train_r, X_test_r, y_test_r, X_train_l, y_train_l, X_test_l, y_test_l

    def _train(self, X_train_r, y_train_r, X_test_r, y_test_r, X_train_l, y_train_l, X_test_l, y_test_l):
        self.model.fit(X_train_r, y_train_r, epochs=50, batch_size=self.batch_size, validation_data=(X_test_r, y_test_r), callbacks=[self.es])
        predictions_right = self.model.predict(y_test_r)

        self.model.fit(X_train_l, y_train_l, epochs=50, batch_size=self.batch_size, validation_data=(X_test_l, y_test_l), callbacks=[self.es])
        predictions_left = self.model.predict(y_test_l)
    
        return predictions_right, predictions_left

    def train_s(self):
        X_train_r, y_train_r, X_test_r, y_test_r, X_train_l, y_train_l, X_test_l, y_test_l = self._prepare_data('s')
        return self._train(X_train_r, y_train_r, X_test_r, y_test_r, X_train_l, y_train_l, X_test_l, y_test_l)
    
    def train_c(self):
        X_train_r, y_train_r, X_test_r, y_test_r, X_train_l, y_train_l, X_test_l, y_test_l = self._prepare_data('c')
        return self._train(X_train_r, y_train_r, X_test_r, y_test_r, X_train_l, y_train_l, X_test_l, y_test_l)

if __name__ == "__main__":

    print('a')