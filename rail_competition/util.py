import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.utils import plot_model

from model_ar import DeepARModel
from model_rnn import RnnModel


class Utility:
    @staticmethod
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def create_array(minus, plus):
        array_c = []
        for m, p in zip(minus, plus):
            if m > 0 and p > 0:
                array_c.append(p)
            elif m < 0 and p < 0:
                array_c.append(m)
            elif abs(m) > abs(p):
                array_c.append(abs(m * p))
            else:
                array_c.append(-1 * abs(m * p))
        return np.array(array_c)


class BaseModelExtension:
    def save_predictions_as_csv(self, predictions_right, predictions_left, columns_right, columns_left, filename="predictions.csv"):
        df_right = pd.DataFrame(predictions_right, columns=columns_right)
        df_left = pd.DataFrame(predictions_left, columns=columns_left)
        
        combined_df = pd.concat([df_right, df_left], axis=1)
        combined_df.to_csv(filename, index=False)

    def print_predictions(self, predictions_right, predictions_left):
        for i in range(len(predictions_right)):
            print(f"Sample {i + 1} - Right: {predictions_right[i]}, Left: {predictions_left[i]}")

    def get_performance_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        return {"MAE": mae, "MSE": mse}

    def print_model_summary(self):
        self.model.summary()

    def plot_model_structure(self, filename="model_structure.png"):
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)


class ExtendedDeepARModel(DeepARModel, BaseModelExtension):

    def save_predictions_as_csv(self, predictions_right, predictions_left, filename="predictions.csv"):
        columns_right = ["YL_M1_B1_W2", "YR_M1_B1_W2"]
        columns_left = ["YL_M1_B1_W1", "YR_M1_B1_W1"]
        super().save_predictions_as_csv(predictions_right, predictions_left, columns_right, columns_left, filename)


class ExtendedRnnModel(RnnModel, BaseModelExtension):

    def save_predictions_as_csv(self, predictions_right, predictions_left, filename="predictions.csv"):
        columns_right = ["YL_M1_B1_W2", "YR_M1_B1_W2"]
        columns_left = ["YL_M1_B1_W1", "YR_M1_B1_W1"]
        super().save_predictions_as_csv(predictions_right, predictions_left, columns_right, columns_left, filename)
