import pandas as pd

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA, GOOG


def std_3(arr, n):
    return pd.Series(arr).rolling(n).std() * 3


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


class MeanReversion(Strategy):
    roll = 50

    def init(self):
        self.he = self.data.Close

        self.he_mean = self.I(SMA, self.he, self.roll)
        self.he_std = self.I(std_3, self.he, self.roll)
        self.he_upper = self.he_mean + self.he_std
        self.he_lower = self.he_mean - self.he_std

        self.he_close = self.I(SMA, self.he, 1)
        
    def next(self):

        if self.he_close < self.he_lower:
            self.buy(
                tp = self.he_mean,
            )

        if self.he_close > self.he_upper:
            self.sell(
                tp = self.he_mean,
            )


def run_sma_cross_strategy_stats(df):
    sma_cross_strategy = Backtest(df, SmaCross, commission=.002,
                                  exclusive_orders=True)
    stats = sma_cross_strategy.run()
    return stats


def run_sma_cross_strategy_plot(df):
    sma_cross_strategy = Backtest(df, SmaCross, commission=.002,
                                  exclusive_orders=True)
    sma_cross_strategy.run()
    return sma_cross_strategy.plot(filename='Оценка стратегии', plot_width=None,
             plot_equity=True, plot_return=True, plot_pl=True,
             plot_volume=True, plot_drawdown=True,
             smooth_equity=False, relative_equity=True,
             resample=True, reverse_indicators=False,
             show_legend=True, open_browser=True)


def run_mean_reversion_stats(df):
    mean_reversion_strategy = Backtest(df, MeanReversion, commission=.002,
                                       exclusive_orders=True)
    stats = mean_reversion_strategy.run()
    return stats


def run_mean_reversion_plot(df):
    mean_reversion_strategy = Backtest(df, MeanReversion, commission=.002,
                                       exclusive_orders=True)
    mean_reversion_strategy.run()
    return mean_reversion_strategy.plot()
