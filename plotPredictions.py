import matplotlib.pyplot as plt
import os
absolute_path = os.path.dirname(__file__)


def plot_predictions(stock, data, training_data_len, predictions):
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
    plt.xlabel('No of Days', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Training Data', 'Validated Data',
                'Predicted Data'], loc='lower right')
    plt.savefig(os.path.join(absolute_path, "assets/chart.png"))
