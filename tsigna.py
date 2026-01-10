#!/usr/bin/env python3

"""
Terminal tool to plot stocks, crypto, pair ratios, and technical indicators.

This is a Python financial analysis tool that runs entirely in the terminal. It
fetches historical stock data from Yahoo Finance, calculates technical
indicators including moving averages, MACD, and RSI, and displays them as text-
based charts using the plotille library. The tool supports single ticker
analysis, ratio comparisons between two tickers, and a special MMRI calculation.
Users can customize the time period and split the terminal display to show
multiple indicators simultaneously.

Copyright (c) 2025 Monsieur Linux

Licensed under the MIT License. See the LICENSE file for details.
"""

# Standard library imports
import argparse
import logging
import math
from pathlib import Path
import requests
import shutil
import sys
import time

# Third-party library imports
import pandas as pd
import plotille
from yahooquery import Ticker  # Alternative fork: ybankinplay

# Configuration constants
CACHE_ENABLE = True
CACHE_PATH = Path.home() / f'.{Path(__file__).stem}'
CACHE_EXPIRY = 300  # 300 seconds = 5 minutes
YEARS_TO_PLOT = 1
INDICATOR_HEIGHT_RATIO = 0.3
MMRI_DIVISOR = 1.61
MOVING_AVG_1 = 20
MOVING_AVG_2 = 50
MOVING_AVG_3 = 200
BB_PERIOD = 20
BB_STD_DEV = 2
MACD_FAST_LEN = 12
MACD_SLOW_LEN = 26
MACD_SIGNAL_LEN = 9
RSI_PERIOD = 14
RSI_OVERBOUGHT_LEVEL = 70
RSI_OVERSOLD_LEVEL = 30
MFI_PERIOD = 14
MFI_OVERBOUGHT_LEVEL = 80
MFI_OVERSOLD_LEVEL = 20
STOCH_K_PERIOD = 14
STOCH_K_SMOOTHING = 3  # Set to 1 for fast stochastics
STOCH_D_PERIOD = 3
STOCH_OVERBOUGHT_LEVEL = 80
STOCH_OVERSOLD_LEVEL = 20
ATR_PERIOD = 14

# Valid terminal plots colors (standard 8-color ANSI palette):
# black, red, green, yellow, blue, magenta, cyan, and white
PRICE_RATIO_COLOR = 'blue'
MOVING_AVG_1_COLOR = 'green'
MOVING_AVG_2_COLOR = 'yellow'
MOVING_AVG_3_COLOR = 'red'
BB_SMA_COLOR = 'cyan'
BB_UPPER_BAND_COLOR = 'red'
BB_LOWER_BAND_COLOR = 'green'
VOLUME_VALUE_COLOR = 'blue'
MACD_VALUE_COLOR = 'blue'
MACD_SIGNAL_COLOR = 'red'
MACD_HISTOGRAM_COLOR = 'green'
RSI_VALUE_COLOR = 'blue'
RSI_OVERBOUGHT_COLOR = 'red'
RSI_OVERSOLD_COLOR = 'green'
MFI_VALUE_COLOR = 'blue'
MFI_OVERBOUGHT_COLOR = 'red'
MFI_OVERSOLD_COLOR = 'green'
STOCH_K_COLOR = 'blue'
STOCH_D_COLOR = 'red'
STOCH_OVERBOUGHT_COLOR = 'red'
STOCH_OVERSOLD_COLOR = 'green'
ATR_VALUE_COLOR = 'blue'

# Get a logger for this script
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ticker1',
                        help='first or only ticker (or special MMRI ticker)')
    parser.add_argument('ticker2', nargs='?', default='',
                        help='second ticker for ratio plot')
    parser.add_argument('-a', '--atr', action='store_true',
                        help='display ATR indicator')
    parser.add_argument('-A', '--atr-only', action='store_true',
                        help='display only ATR indicator')
    parser.add_argument('-b', '--bollinger', action='store_true',
                        help='display Bollinger Bands indicator')
    parser.add_argument('-f', '--mfi', action='store_true',
                        help='display MFI indicator')
    parser.add_argument('-F', '--mfi-only', action='store_true',
                        help='display only MFI indicator')
    parser.add_argument('-m', '--macd', action='store_true',
                        help='display MACD indicator')
    parser.add_argument('-M', '--macd-only', action='store_true',
                        help='display only MACD indicator')
    parser.add_argument('-n', '--no-cache', action='store_true',
                        help='bypass cache and get latest data')
    parser.add_argument('-r', '--rsi', action='store_true',
                        help='display RSI indicator')
    parser.add_argument('-R', '--rsi-only', action='store_true',
                        help='display only RSI indicator')
    parser.add_argument('-s', '--stoch', action='store_true',
                        help='display Stochastics indicator')
    parser.add_argument('-S', '--stoch-only', action='store_true',
                        help='display only Stochastics indicator')
    parser.add_argument('-v', '--volume', action='store_true',
                        help='display volume')
    parser.add_argument('-V', '--volume-only', action='store_true',
                        help='display only volume')
    parser.add_argument('-y', '--years', type=int, default=YEARS_TO_PLOT,
                        help='set years to plot, use 0 for ytd')
    args = parser.parse_args()
    
    ticker1, ticker2, plot_name = get_tickers_and_plot_name(args)
    main_ind = 'bb' if args.bollinger else 'ma'
    xtra_ind = []
    if args.volume or args.volume_only: xtra_ind.append('vol')
    if args.macd or args.macd_only: xtra_ind.append('macd')
    if args.rsi or args.rsi_only: xtra_ind.append('rsi')
    if args.mfi or args.mfi_only: xtra_ind.append('mfi')
    if args.stoch or args.stoch_only: xtra_ind.append('stoch')
    if args.atr or args.atr_only: xtra_ind.append('atr')

    try:
        df1, df2 = get_data(ticker1, ticker2, no_cache=args.no_cache)
    except KeyError as e:
        logger.error(f'Invalid ticker: {e}')
    except requests.exceptions.RequestException as e:
        logger.error(f'Connection failed: {e}')
    except AssertionError as e:
        logger.error(f'Assert failed: {e}')
    except Exception as e:
        logger.exception(f'Unexpected error: {e}')
    else:
        df = process_data(df1, df2, args.years, plot_name, main_ind, xtra_ind)
        if ticker2 != '' and (args.mfi or args.mfi_only):
            logger.error(f'MFI indicator not available for ratio plot')
        elif ticker2 != '' and (args.stoch or args.stoch_only):
            logger.error(f'Stochastics indicator not available for ratio plot')
        elif ticker2 != '' and (args.atr or args.atr_only):
            logger.error(f'ATR indicator not available for ratio plot')
        elif args.volume_only:
            plot_data(df, plot_name, 'vol')
        elif args.macd_only:
            plot_data(df, plot_name, 'macd')
        elif args.rsi_only:
            plot_data(df, plot_name, 'rsi')
        elif args.mfi_only:
            plot_data(df, plot_name, 'mfi')
        elif args.stoch_only:
            plot_data(df, plot_name, 'stoch')
        elif args.atr_only:
            plot_data(df, plot_name, 'atr')
        else:
            num_ind = len(xtra_ind)
            if num_ind > 2:
                logger.error(f'A maximum of two indicators can be displayed')
            elif num_ind == 2:
                plot_data(df, plot_name, main_ind, 1-2*INDICATOR_HEIGHT_RATIO)
                plot_data(df, plot_name, xtra_ind[0], INDICATOR_HEIGHT_RATIO)
                plot_data(df, plot_name, xtra_ind[1], INDICATOR_HEIGHT_RATIO)
            elif num_ind == 1:
                plot_data(df, plot_name, main_ind, 1-INDICATOR_HEIGHT_RATIO)
                plot_data(df, plot_name, xtra_ind[0], INDICATOR_HEIGHT_RATIO)
            else:
                plot_data(df, plot_name, main_ind)


def get_tickers_and_plot_name(args):
    ticker1 = args.ticker1.lower()
    ticker2 = args.ticker2.lower()

    if ticker2 != '':
        # Plot the ratio between ticker1 and ticker2
        plot_name = ticker1.upper() + ' vs ' + ticker2.upper()
    else:
        plot_name = ticker1.upper()
        if ticker1 == 'mmri':
            # Special "ticker" to plot the Mannarino Market Risk Indicator
            ticker1 = 'dx=f'
            ticker2 = '^tnx'  # '10y=f' has no historical data
            plot_name = 'MMRI'

    return ticker1, ticker2, plot_name


def get_data(ticker1, ticker2, no_cache=False):
    fetch_data = True
    df2 = pd.DataFrame()
    CACHE_PATH.mkdir(parents=True, exist_ok=True)
    path1 = Path(f'{CACHE_PATH}/{ticker1}.csv')
    path2 = Path(f'{CACHE_PATH}/{ticker2}.csv')

    if CACHE_ENABLE and not no_cache:
        fetch_data = False
        now = time.time()

        if path1.is_file() and (now - path1.stat().st_mtime < CACHE_EXPIRY):
            logger.info(f'Getting {ticker1} data from cache')
            df1 = pd.read_csv(path1, parse_dates=['date'])
            df1.set_index('date', inplace=True)
        else:
            fetch_data = True
        
        if ticker2 != '':
            if path2.is_file() and (now - path2.stat().st_mtime < CACHE_EXPIRY):
                logger.info(f'Getting {ticker2} data from cache')
                df2 = pd.read_csv(path2, parse_dates=['date'])
                df2.set_index('date', inplace=True)
            else:
                fetch_data = True

    if fetch_data:
        logger.info('Getting ticker(s) data from Yahoo Finance')
        tickers = [ticker1, ticker2] if ticker2 != '' else [ticker1]       
        tickers = Ticker(tickers)

        df = tickers.history(period='10y', interval='1d')
        df1 = df.loc[ticker1]
        if CACHE_ENABLE: df1.to_csv(path1, index=True)

        if ticker2 != '':
            df2 = df.loc[ticker2]
            if CACHE_ENABLE: df2.to_csv(path2, index=True)

    # Make sure all dates have the same format (remove time from last date)
    # normalize() sets the time to midnight while keeping pandas dates types
    df1.index = pd.to_datetime(df1.index,utc=True,format='ISO8601').normalize()
    df2.index = pd.to_datetime(df2.index,utc=True,format='ISO8601').normalize()
    assert df1.index.is_unique, f'Duplicate date for {ticker1}'
    assert df2.index.is_unique, f'Duplicate date for {ticker2}'
    df1 = df1.groupby(df1.index).last()  # Make sure there are no duplicates
    df2 = df2.groupby(df2.index).last()

    return df1, df2


def process_data(df1, df2, years, plot_name, main_ind, xtra_ind):
    if df2.empty:
        # Only one ticker has been provided, so this is the data to plot
        df = df1
    else:
        # Two tickers has been provided, so compute the ratios between them
        dates = []
        values = []

        # TODO : Iterating through the index is not idiomatic pandas. A better
        # way would be to align the two DF and perform a vectorized operation.
        for date in df1.index.unique():
            if date in df2.index:
                # .at is better than .loc for single value
                value1 = df1.at[date, 'adjclose']
                value2 = df2.at[date, 'adjclose']
            
                if value1 > 0 and value2 > 0:
                    dates.append(date)
                    if plot_name == 'MMRI':
                        # mmri = dx * 10y / 1.61
                        values.append(value1 * value2 / MMRI_DIVISOR)
                    else:
                        values.append(value1 / value2)

        # Create the pair dataframe
        df = pd.DataFrame({ 'date': dates, 'adjclose': values })
        df.set_index('date', inplace=True)

    # Calculate and add columns for requested indicators
    if 'ma' in main_ind: df = add_moving_averages(df)
    if 'bb' in main_ind: df = add_bollinger_bands(df)
    if 'macd' in xtra_ind: df = add_macd(df)
    if 'rsi' in xtra_ind: df = add_rsi(df)

    if 'low' in df.columns:
        # Indicators N/A for ratio plots (OHLC prices and/or volume required)
        if 'mfi' in xtra_ind: df = add_mfi(df)
        if 'stoch' in xtra_ind: df = add_stochastics(df)
        if 'atr' in xtra_ind: df = add_atr(df)

    # Keep only the data range to be plotted (use pandas dates types)
    today = pd.Timestamp.now(tz='UTC').normalize()
    
    if years == 0:
        start_day = today.replace(month=1, day=1)  # ytd plot
    else:
        start_day = today.replace(year=today.year - years)

    df = df[df.index >= start_day]
    
    logger.debug(f'today is {today}')
    logger.debug(f'start_day is {start_day}')

    return df


def plot_data(df, plot_name, plot_type, height_ratio=1):
    # Display the plot in the terminal
    dates = df.index.tolist()
    
    if plot_type == 'vol':
        volume = df['volume'].tolist()
        all_values = volume
    elif plot_type == 'macd':
        macd = df['macd'].tolist()
        signal = df['signal'].tolist()
        histogram = df['histogram'].tolist()
        all_values = macd + signal + histogram
    elif plot_type == 'rsi':
        rsi = df['rsi'].tolist()
        overbought = [RSI_OVERBOUGHT_LEVEL] * len(dates)
        oversold = [RSI_OVERSOLD_LEVEL] * len(dates)
        all_values = rsi + overbought + oversold
    elif plot_type == 'mfi':
        mfi = df['mfi'].tolist()
        overbought = [MFI_OVERBOUGHT_LEVEL] * len(dates)
        oversold = [MFI_OVERSOLD_LEVEL] * len(dates)
        all_values = mfi + overbought + oversold
    elif plot_type == 'stoch':
        stoch_k = df['stoch_k'].tolist()
        stoch_d = df['stoch_d'].tolist()
        overbought = [STOCH_OVERBOUGHT_LEVEL] * len(dates)
        oversold = [STOCH_OVERSOLD_LEVEL] * len(dates)
        all_values = stoch_k + stoch_d + overbought + oversold
    elif plot_type == 'atr':
        atr = df['atr'].tolist()
        all_values = atr
    elif plot_type == 'bb':
        close = df['adjclose'].tolist()
        sma = df['sma'].tolist()
        upper = df['upper'].tolist()
        lower = df['lower'].tolist()
        all_values = close + sma + upper + lower
    else:  # Main plot with moving averages
        close = df['adjclose'].tolist()
        ma1 = df['ma1'].tolist()
        ma2 = df['ma2'].tolist()
        ma3 = df['ma3'].tolist()
        all_values = close + ma1 + ma2 + ma3
        
    fig = plotille.Figure()

    # Determine the dimensions and limits of the plot
    fig.width = shutil.get_terminal_size()[0] - 21
    fig.height = math.floor(shutil.get_terminal_size()[1] * height_ratio) - 5
    fig.set_x_limits(dates[0], dates[-1])
    fig.set_y_limits(min(all_values), max(all_values))

    # Prepare the plots and text to display
    if plot_type == 'vol':
        fig.plot(dates, volume, lc=VOLUME_VALUE_COLOR)
        last = f'{volume[-1]:.0f}'
        text = f'Volume last value: {last}'
    elif plot_type == 'macd':
        fig.plot(dates, signal, lc=MACD_SIGNAL_COLOR)
        fig.plot(dates, macd, lc=MACD_VALUE_COLOR)
        fig.plot(dates, histogram, lc=MACD_HISTOGRAM_COLOR)
        last = f'{histogram[-1]:.2f}'
        text = f'MACD histogram last value: {last}'
    elif plot_type == 'rsi':
        fig.plot(dates, overbought, lc=RSI_OVERBOUGHT_COLOR)
        fig.plot(dates, oversold, lc=RSI_OVERSOLD_COLOR)
        fig.plot(dates, rsi, lc=RSI_VALUE_COLOR)
        last = f'{rsi[-1]:.2f}'
        text = f'RSI last value: {last}'
    elif plot_type == 'mfi':
        fig.plot(dates, overbought, lc=MFI_OVERBOUGHT_COLOR)
        fig.plot(dates, oversold, lc=MFI_OVERSOLD_COLOR)
        fig.plot(dates, mfi, lc=MFI_VALUE_COLOR)
        last = f'{mfi[-1]:.2f}'
        text = f'MFI last value: {last}'
    elif plot_type == 'stoch':
        fig.plot(dates, overbought, lc=STOCH_OVERBOUGHT_COLOR)
        fig.plot(dates, oversold, lc=STOCH_OVERSOLD_COLOR)
        fig.plot(dates, stoch_k, lc=STOCH_K_COLOR)
        fig.plot(dates, stoch_d, lc=STOCH_D_COLOR)
        last = f'{stoch_d[-1]:.2f}'
        text = f'Stochastics last value: {last}'
    elif plot_type == 'atr':
        fig.plot(dates, atr, lc=ATR_VALUE_COLOR)
        last = f'{atr[-1]:.2f}'
        text = f'ATR last value: {last}'
    elif plot_type == 'bb':
        fig.plot(dates, close, lc=PRICE_RATIO_COLOR)
        fig.plot(dates, sma, lc=BB_SMA_COLOR)
        fig.plot(dates, upper, lc=BB_UPPER_BAND_COLOR)
        fig.plot(dates, lower, lc=BB_LOWER_BAND_COLOR)
        last = f'{close[-1]:.0f}' if close[-1] > 1000 else f'{close[-1]:.2f}'
        change = f'{(close[-1] / close[0] - 1) * 100:+.0f}'
        text = f'{plot_name} last value: {last} ({change}%)'
    else:  # Main plot with moving averages
        fig.plot(dates, ma3, lc=MOVING_AVG_3_COLOR)
        fig.plot(dates, ma2, lc=MOVING_AVG_2_COLOR)
        fig.plot(dates, ma1, lc=MOVING_AVG_1_COLOR)
        fig.plot(dates, close, lc=PRICE_RATIO_COLOR)
        last = f'{close[-1]:.0f}' if close[-1] > 1000 else f'{close[-1]:.2f}'
        change = f'{(close[-1] / close[0] - 1) * 100:+.0f}'
        text = f'{plot_name} last value: {last} ({change}%)'

    # Display the last value text
    x = dates[0] + (dates[-1] - dates[0]) * 0.55
    y = min(all_values)
    fig.text([x], [y], [text])

    print(fig.show(legend=False))


def add_moving_averages(df):
    # Calculate and add moving averages
    df = df.copy()
    df['ma1'] = df['adjclose'].rolling(window=MOVING_AVG_1).mean()
    df['ma2'] = df['adjclose'].rolling(window=MOVING_AVG_2).mean()
    df['ma3'] = df['adjclose'].rolling(window=MOVING_AVG_3).mean()
    df = df.fillna(0)
    return df
    

def add_macd(df):
    # Calculate and add MACD indicator (Moving Average Convergence Divergence)
    df = df.copy()
    fast = df['adjclose'].ewm(span=MACD_FAST_LEN, adjust=False).mean()
    slow = df['adjclose'].ewm(span=MACD_SLOW_LEN, adjust=False).mean()
    df['macd'] = fast - slow
    df['signal'] = df['macd'].ewm(span=MACD_SIGNAL_LEN, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    return df
    

def add_rsi(df):
    # Calculate and add RSI indicator (Relative Strength Index)
    # Calculate the average gain and average loss using Wilder's Smoothing
    # We use a 'com' span of period-1 to match the standard RSI calculation
    df = df.copy()
    delta = df['adjclose'].diff()       # Difference from the previous day
    gain = delta.where(delta > 0, 0)    # Keep gains and replace losses with 0
    loss = -delta.where(delta < 0, 0)   # keep -losses and replace gains with 0
    avg_gain = gain.ewm(com=RSI_PERIOD-1, adjust=False).mean()  # Average gain
    avg_loss = loss.ewm(com=RSI_PERIOD-1, adjust=False).mean()  # Average loss
    rs = avg_gain / (avg_loss + 1e-10)  # RS (avoid division by zero)
    df['rsi'] = 100 - (100 / (1 + rs))  # RSI (normalize to a scale of 0 to 100)
    return df


def add_mfi(df):
    # Calculate and add MFI indicator (Money Flow Index)
    df = df.copy()
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    delta = typical_price.diff()  # Difference from the previous day
    pos_mf = money_flow.where(delta > 0, 0)  # Positive money flow
    neg_mf = money_flow.where(delta < 0, 0)  # Negative money flow
    avg_pos_mf = pos_mf.rolling(window=MFI_PERIOD, min_periods=1).mean()
    avg_neg_mf = neg_mf.rolling(window=MFI_PERIOD, min_periods=1).mean()
    mfr = avg_pos_mf / (avg_neg_mf + 1e-10)  # Avoid division by zero
    df['mfi'] = 100 - (100 / (1 + mfr))  # Normalize to a scale of 0 to 100
    df['mfi'] = df['mfi'].fillna(100)  # Fill NaN values
    return df


def add_stochastics(df):
    # Calculate and add Stochastic Oscillator indicator
    df = df.copy()
    low_min = df['low'].rolling(window=STOCH_K_PERIOD).min()    # Lowest low
    high_max = df['high'].rolling(window=STOCH_K_PERIOD).max()  # Highest high
    fast_k = ((df['close'] - low_min) / (high_max - low_min)) * 100  # Fast
    df['stoch_k'] = fast_k.rolling(window=STOCH_K_SMOOTHING).mean()  # Smoothed
    df['stoch_d'] = df['stoch_k'].rolling(window=STOCH_D_PERIOD).mean() # Slow
    return df


def add_bollinger_bands(df):
    # Calculate and add Bollinger Bands indicator
    df = df.copy()
    df['sma'] = df['adjclose'].rolling(window=BB_PERIOD).mean()  # Rolling mean
    std = df['adjclose'].rolling(window=BB_PERIOD).std() # Rolling std deviation
    df['upper'] = df['sma'] + (std * BB_STD_DEV)  # Upper band
    df['lower'] = df['sma'] - (std * BB_STD_DEV)  # Lower band
    return df


def add_atr(df):
    # Calculate and add ATR indicator (Average True Range)
    df = df.copy()
    tr1 = df['high'] - df['low']                        # high - low
    tr2 = (df['high'] - df['close'].shift()).abs()      # high - previous close
    tr3 = (df['low'] - df['close'].shift()).abs()       # low - previous close
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1) # Max of the 3 components
    df['atr'] = tr.ewm(com=ATR_PERIOD-1, adjust=False).mean() # Wilder's Smoothing
    return df


def log_data_frame(df, description):
    """ This function is used only for debugging. """
    logger.debug(f'DataFrame {description}\n{df}')
    #logger.debug(f'DataFrame index data type: {df.index.dtype}')
    #logger.debug(f'DataFrame index class: {type(df.index)}')
    #logger.debug(f'DataFrame columns data types\n{df.dtypes}')
    #logger.debug(f'DataFrame statistics\n{df.describe()}')  # Mean, min, max...
    sys.exit()


if __name__ == '__main__':
    # Configure the root logger
    logging.basicConfig(level=logging.WARNING,
                        format='%(levelname)s - %(message)s')
    
    # Configure this script's logger
    #logger.setLevel(logging.DEBUG)

    main()
