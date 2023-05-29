import matplotlib.pyplot as plt



class Visualizations:
    
    def plot_price_diff(df, y, labels, colors):
        for i in range(len(y)):
            plt.plot(df.index,df[y[i]], label = labels[i], color = colors[i], ls='-.')
        
        plt.ylabel('Stock price')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.title('Change in Price')
        plt.legend()
        plt.show()
    
    
    def bollinger_bands_visualization(df, price_col, UB, LB, col_buy, col_sell):
        fig, ax = plt.subplots(figsize = (15,10))
        plt.title('Bollinger bands and trading strategy')
        plt.ylabel('Price')
        plt.xlabel('Date')
        ax.plot(df[price_col], label = 'Price', alpha = 0.5, color = 'blue', linewidth = 2.5)
        ax.plot(df[UB], label = 'Upper Band', alpha = 0.5, color = 'red', linewidth = 2.5 , linestyle = '--')
        ax.plot(df[LB], label = 'Lower Band', alpha = 0.5, color = 'purple', linewidth = 2.5 , linestyle = '--')
        # ax.fill_between(forecast.index, forecast['UpperBand'], forecast['LowerBand'], color = 'grey')
        ax.scatter(df.index, df[col_buy], label = 'Buy',alpha = 1, marker = '^', color = 'green')
        ax.scatter(df.index, df[col_sell], label = 'Sell', alpha = 1, marker = 'v', color = 'red')
        plt.legend()
        plt.show()