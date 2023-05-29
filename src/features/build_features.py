import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler

    
class Features:
    
    def test_stationary(df):

        # determining rolling statistics
        rolmean = pd.Series(df).rolling(window=7).mean()
        rolstd = pd.Series(df).rolling(window=7).std()

        # plotting rolling statistics:
        orig = plt.plot(df, color='blue', label = 'original')
        mean = plt.plot(rolmean, color='red', label = 'Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling std')
        plt.legend(loc = 'best')
        plt.title('Rolling mean & standard deviation')
        plt.xticks(rotation=45)
        plt.show(block = False)


        # perform dickey fuller test
        print('Results of Dickey Fuller Test: ')
        dftest = adfuller(df, autolag = 'AIC')
        result = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value','# laags used', 'Number of observations used'])
        for key, value in dftest[4].items():
            result['Critical value (%s)' %key] = value
        print(result)

        if result['p-value'] <= 0.05:
            print('strong evidence against null hypothesis, reject the null hypothesis. Data has no unit and is stationary')
        else:
            print('week evidence against null hypothesis, time series has a unit root, indicating it is non-stationary')

    def preprocessing(train_df, test_df, sc):

        train_array = np.array(train_df).reshape(-1, 1)
        test_array = np.array(test_df).reshape(-1, 1)
        scaled_train = sc.fit_transform(train_array)
        scaled_test = sc.transform(test_array)
        return sc, scaled_train, scaled_test