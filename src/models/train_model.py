from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from prophet import Prophet
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

    
class Models:
    
    def auto_arima_model(data):
        model_auto_arima = auto_arima(data, test = 'adf', stepwise = False, trace = True, seasonal=False, suppress_warnings = True)
        print(model_auto_arima.summary())

        model_auto_arima.plot_diagnostics(figsize = (12, 10))
        plt.show()

        best_p, best_d, best_q = model_auto_arima.order
        print("Best (p, d, q):", best_p, best_d, best_q)


    def arima_model(train_data, test_data, p, d, q):
        arima = ARIMA(train_data, order = (p, d, q))
        arima_model_fit = arima.fit()
        print(arima_model_fit.summary())

        plt.plot(test_data, color = 'green')
        plt.plot(arima_model_fit.fittedvalues, color = 'blue')
        plt.legend()

        return arima_model_fit


    def model_sarimax(df, p, d, q, s):
        sarimax_model = SARIMAX(df, order = (p, d, q), seasonal_order = (p, d, q, s))
        sarimax_model_fit = sarimax_model.fit()
        print(sarimax_model_fit.summary())
        return sarimax_model_fit


    def prophet_model(train_df, test_df, periods):
        model_prophet = Prophet()
        model_prophet.fit(train_df)

        forecast = model_prophet.predict(test_df)
        print(forecast.head())
        return model_prophet, forecast


    def lstm_model(input, features):
        model = Sequential()
        model.add(LSTM(100, activation = 'relu', input_shape = (input, features)))
        model.add(Dense(1))
        model.compile(optimizer = 'adam', loss = 'mae')
        return model
