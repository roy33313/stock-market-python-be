from keras import Sequential
from keras.layers import Dense, LSTM


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
