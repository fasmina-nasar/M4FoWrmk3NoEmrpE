import numpy as np    
    
class Forecast: 

    def predictions(df, train_data, test_data, col, model):

        start_index = len(train_data)
        end_index = len(train_data)+len(test_data)-1

        forecast = model.predict(start = start_index, end = end_index)
        forecast.index = df.index[start_index : end_index+1]

        return forecast


    def bollinger_bands(data, col, window_size):
        rolling_mean = data[col].rolling(window = window_size).mean()
        rolling_std = data[col].rolling(window = window_size).std()
        data['UpperBand'] = rolling_mean + (2 * rolling_std)
        data['LowerBand'] = rolling_mean - (2 * rolling_std)
        return data


    def bb_strategy(data, col):
        buy_price = []
        sell_price = []
        for i in range(len(data)):
            if data[col][i] < data['LowerBand'][i]:
                buy_price.append(data[col][i])
                sell_price.append(np.nan)
            elif data[col][i] > data['UpperBand'][i]:
                sell_price.append(data[col][i])
                buy_price.append(np.nan)
            else:
                sell_price.append(np.nan)
                buy_price.append(np.nan)
        return buy_price, sell_price