from flask import *
import os
import pandas as pd
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
from keras.models import load_model
from easygui import *
import tensorflow as tf
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predrop():
    data = request.form['comment']
    data1 = request.form['comment1']
    dataframes = []
    stock_name = []

    data22 = data + '.csv'
    data33 = 'C:/Users/royal/OneDrive/Desktop/Stock-Market-Prediction/stock_market_prediction-main/stock_market_prediction-main/datasets/' + data22
    dataframes = pd.read_csv(data33)
    stock_name = data
    dataframes['Adj Close'].plot()
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {stock_name[0]}")

    dataframes['Volume'].plot()
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Volume of {stock_name[0]}")
    plt.savefig(
        "somefig.png")

    # plt.show()

    # %%
    ma_day = [10, 50, 100, 365]
    for ma in ma_day:
        column_name = f"MA for {ma} days"
        print('column_name', column_name)
        dataframes[column_name] = dataframes['Adj Close'].rolling(
            ma).mean()
    dataframes['Daily Return'] = dataframes['Adj Close'].pct_change()

    def build_training_dataset(input_ds):
        input_ds.reset_index()
        data = input_ds.filter(items=['Close'])
        dataset = data.values
        training_data_len = int(np.ceil(len(dataset) * .95))
        return data, dataset, training_data_len

    # Test the function
    training_data_df, training_dataset_np, training_data_len = build_training_dataset(
        dataframes)
    dataset = training_dataset_np
    data = training_data_df

    def scale_the_data(dataset):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        return scaler, scaled_data

    # Test the function
    scaler, scaled_data = scale_the_data(training_dataset_np)

    # Create the training data set
    # Create the scaled training data set

    def split_train_dataset(training_data_len):
        train_data = scaled_data[0:int(training_data_len), :]
        # Split the data into x_train and y_train data sets
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
            if i <= 61:
                print('.')

        # Convert to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data
        x_train = np.reshape(
            x_train, (x_train.shape[0], x_train.shape[1], 1))
        # x_train.shape
        return x_train, y_train

    # Test the function
    x_train, y_train = split_train_dataset(training_data_len)

    # %%
    def build_lstm_model(x_train, y_train):
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True,
                       input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        # adam ~ Stochastic Gradient descent method.
        model.compile(optimizer='adam', loss='mean_squared_error')
    
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        return model

        # Test the function

    lstm_model = build_lstm_model(x_train, y_train)

    lstm_model.save('model.h5')

    def create_testing_data_set(model, scaler, training_data_len, test_data_len):
        # Create the testing data set
        # Create a new array containing scaled values from index 1543 to 2002
        test_data = scaled_data[training_data_len - test_data_len:, :]
        # Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(test_data_len, len(test_data)):
            x_test.append(test_data[i - test_data_len:i, 0])

        # Convert the data to a numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Get the models predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        predicted_main_out = np.mean(predictions)

        # Get the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        return (x_test, y_test, predictions, rmse, predicted_main_out)

    # Test the function
    TEST_DATA_LENGTH = 60
    print('reach4')
    x_test, y_test, predictions, rmse, predicted_main_out = create_testing_data_set(lstm_model, scaler, training_data_len,
                                                                                    TEST_DATA_LENGTH)

    # %%

    def plot_predictions(stock, data, training_data_len):
        # Plot the data
        print('stockname', stock)
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Visualize the data
        plt.figure(figsize=(16, 6))
        title = stock + ' Model Forecast'
        ylabel = stock + ' Close Price'
        plt.title(title)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Training Data', 'Validated Data',
                    'Predicted Data'], loc='lower right')
        plt.savefig(
            "mainImage3.png")

        # plt.show()

        return valid

    # Test the function
    valid = plot_predictions(stock_name, data, training_data_len)
    print('valid', valid)

    response = predicted_main_out
    predicted_main_out = predicted_main_out.astype(float)
    data_dict = {"predicted_main_out": predicted_main_out.tolist()}
    return str(data_dict)


@app.route('/predictinfo/getImage')
def get_image():
    image_filename = 'graph.png'
    return send_file(image_filename, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
