from buildTrainingDataset import build_training_dataset
from splitTrainDataset import split_train_dataset
from buildLstmModel import build_lstm_model
from plotPredictions import plot_predictions

from flask_cors import CORS
import tensorflow as tf
from easygui import *
from keras.models import load_model

from flask import *
import os
import pandas as pd
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, Flatten
from sklearn.preprocessing import MinMaxScaler
import numpy as np  # linear algebra
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)
CORS(app)


absolute_path = os.path.dirname(__file__)
relative_path = "src/lib"


@app.route('/predict', methods=['POST'])
def func():
    jsonData = request.get_json()
    stock_name = jsonData["stock"]

    dataframes = pd.read_csv('datasets/'+stock_name+'.csv')
    training_data_df, training_dataset_np, training_data_len = build_training_dataset(
        dataframes)
    dataset = training_dataset_np
    data = training_data_df

    def scale_the_data(dataset):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        return scaler, scaled_data

    scaler, scaled_data = scale_the_data(training_dataset_np)
    x_train, y_train = split_train_dataset(training_data_len, scaled_data)
    lstm_model = build_lstm_model(x_train, y_train)

    def create_testing_data_set(model, scaler, training_data_len, test_data_len):
        test_data = scaled_data[training_data_len - test_data_len:, :]

        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(test_data_len, len(test_data)):
            x_test.append(test_data[i - test_data_len:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        predicted_main_out = np.mean(predictions)

        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        return (x_test, y_test, predictions, rmse, predicted_main_out)

    x_test, y_test, predictions, rmse, predicted_main_out = create_testing_data_set(lstm_model, scaler, training_data_len,
                                                                                    60)

    plot_predictions(stock_name, data, training_data_len, predictions)
    predicted_main_out = predicted_main_out.astype(float)
    data_dict = {"predicted_main_out": predicted_main_out.tolist()}
    return jsonify(data_dict)


@app.route('/predict/getImage')
def get_image():
    image_filename = os.path.join(
        absolute_path, 'assets/')+request.args.get("graph")+'.png'
    return send_file(image_filename, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
