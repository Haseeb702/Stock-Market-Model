import requests
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def stock_predict(stock):
    #url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock}&apikey=Y0ZWS1FQO8I2RYPY&datatype=csv&outputsize=full'
    #request json
    #response = requests.get(url) 
    # do error catching with .request == 200
    #df = pd.read_csv(io.StringIO(response.text)) #converts csv into pandas df

    df = pd.read_csv('daily_AAPL.csv')
    df = df.iloc[::-1]

    close_data = df.filter(['close'])
    dataset = close_data.values # a 2d array of the close data
    training = int(np.ceil(len(dataset) * 0.97))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    train_data = scaled_data[0:int(training), :] 

    x_train = []
    y_train = []

    for i in range(500, len(train_data)): 
        x_train.append(train_data[i-500:i, 0]) 
        y_train.append(train_data[i, 0]) 
    
    x_train, y_train = np.array(x_train), np.array(y_train) 
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(x_train.shape[1], 1)))
    model.add(keras.layers.LSTM(units=32, return_sequences=True))
    model.add(keras.layers.LSTM(units=32))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))

    optimizer = keras.optimizers.Adam(clipvalue=1.0)
    model.compile(optimizer=optimizer, loss ='mean_squared_error')
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, 
                        validation_split=0.2, callbacks=[early_stop], verbose=1)
    
    # Making test data to compare
    test_data = scaled_data[training - 500:, :] 

    # Use the last 60 days of test data
    last_60_days = test_data.copy()

    # New predictions
    future_predictions = []

    for i in range(300):  # Predict for 186 days (approx. 6 months)
        print(f"Step {i+1}: last_60_days shape = {last_60_days.shape}")

        # Reshape last 60 days for prediction
        try:
            input_data = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
        except Exception as e:
            print(f"Error during reshape: {e}")
            print(f"last_60_days: {last_60_days}")
            raise

        print(f"Step {i+1}: input_data shape = {input_data.shape}")

        # Predict the next day
        new_pred = model.predict(input_data)

        # Store the new prediction
        future_predictions.append(new_pred[0, 0])

        # Update last_60_days to include the latest prediction
        new_pred_reshaped = new_pred.reshape(1, 1)  # Ensure it's 2D for stacking
        last_60_days = np.vstack([last_60_days[1:], new_pred_reshaped])

    # Inverse transform to get the original scale
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)


    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(dataset)), dataset, label='Historical Data')
    plt.plot(np.arange(len(dataset), len(dataset) + len(future_predictions)),
            future_predictions, label='Future Predictions', color='red')
    plt.xlabel('Days')
    plt.ylabel('Close Price')
    plt.title('Stock Price Predictions')
    plt.legend()
    plt.show()
    
    return 1


stock_predict('IBM')