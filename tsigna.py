#!/usr/bin/env python3

"""
A terminal-based financial charting tool for plotting stock prices, moving
averages, and technical indicators. It is most useful for medium-term trading.

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
MOVING_AVG_1 = 20
MOVING_AVG_2 = 50
MOVING_AVG_3 = 200
YEARS_TO_PLOT = 1
HEIGHT_RATIO = 0.3
MMRI_DIVISOR = 1.61
MACD_FAST_LEN = 12
MACD_SLOW_LEN = 26
MACD_SIGNAL_LEN = 9
RSI_PERIOD = 14
RSI_OVERBOUGHT_LEVEL = 70
RSI_OVERSOLD_LEVEL = 30

# Valid terminal plots colors (standard 8-color ANSI palette):
# black, red, green, yellow, blue, magenta, cyan, and white
PRICE_RATIO_COLOR = 'blue'
MOVING_AVG_1_COLOR = 'green'
MOVING_AVG_2_COLOR = 'yellow'
MOVING_AVG_3_COLOR = 'red'
MACD_VALUE_COLOR = 'blue'
MACD_SIGNAL_COLOR = 'red'
MACD_HISTOGRAM_COLOR = 'green'
RSI_VALUE_COLOR = 'blue'
RSI_OVERBOUGHT_COLOR = 'red'
RSI_OVERSOLD_COLOR = 'green'

# Get a logger for this script
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('ticker1',
                        help='first or only ticker (or special MMRI ticker)')
    parser.add_argument('ticker2', nargs='?', default='',
                        help='second ticker for ratio plot')
    parser.add_argument('-m', '--macd', action='store_true',
                        help='display MACD indicator plot')
    parser.add_argument('-M', '--macd-only', action='store_true', dest='macd_only',
                        help='display only MACD indicator plot')
    parser.add_argument('-n', '--no-cache', action='store_true', dest='no_cache',
                        help='bypass cache and get latest data')
    parser.add_argument('-p', '--periods', nargs=3, type=int,
                        default=[MOVING_AVG_1, MOVING_AVG_2, MOVING_AVG_3],
                        help='set moving averages periods')
    parser.add_argument('-r', '--rsi', action='store_true',
                        help='display RSI indicator plot')
    parser.add_argument('-R', '--rsi-only', action='store_true', dest='rsi_only',
                        help='display only RSI indicator plot')
    parser.add_argument('-y', '--years', type=int, default=YEARS_TO_PLOT,
                        help='set years to plot, use 0 for ytd')
    args = parser.parse_args()
    
    ticker1, ticker2, plot_name = get_tickers_and_plot_name(args)

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
        df = process_data(df1, df2, args.periods, args.years, plot_name)
        if args.macd_only:
            plot_data(df, plot_name, 'macd')
        elif args.rsi_only:
            plot_data(df, plot_name, 'rsi')
        elif args.macd and args.rsi:
            plot_data(df, plot_name, 'main', 1-2*HEIGHT_RATIO)
            plot_data(df, plot_name, 'macd', HEIGHT_RATIO)
            plot_data(df, plot_name, 'rsi', HEIGHT_RATIO)
        elif args.macd:
            plot_data(df, plot_name, 'main', 1-HEIGHT_RATIO)
            plot_data(df, plot_name, 'macd', HEIGHT_RATIO)
        elif args.rsi:
            plot_data(df, plot_name, 'main', 1-HEIGHT_RATIO)
            plot_data(df, plot_name, 'rsi', HEIGHT_RATIO)
        else:
            plot_data(df, plot_name, 'main')


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
    df1.index = pd.to_datetime(df1.index, utc=True, format='ISO8601').normalize()
    df2.index = pd.to_datetime(df2.index, utc=True, format='ISO8601').normalize()
    assert df1.index.is_unique, f'Duplicate date for {ticker1}'
    assert df2.index.is_unique, f'Duplicate date for {ticker2}'
    df1 = df1.groupby(df1.index).last()  # Make sure there are no duplicates
    df2 = df2.groupby(df2.index).last()

    return df1, df2


def process_data(df1, df2, periods, years, plot_name):
    if df2.empty:
        # Only one ticker has been provided, so this is the data to plot
        df = df1
    else:
        # Compute the ratios between the two tickers
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

    # Create new columns for the moving averages
    df['ma1'] = df['adjclose'].rolling(window=periods[0]).mean()
    df['ma2'] = df['adjclose'].rolling(window=periods[1]).mean()
    df['ma3'] = df['adjclose'].rolling(window=periods[2]).mean()
    df.fillna(0, inplace=True)
    
    # Create new columns for the MACD indicator
    fast = df['adjclose'].ewm(span=MACD_FAST_LEN, adjust=False).mean()
    slow = df['adjclose'].ewm(span=MACD_SLOW_LEN, adjust=False).mean()
    df['macd'] = fast - slow
    df['signal'] = df['macd'].ewm(span=MACD_SIGNAL_LEN, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    
    # Create a new column for the RSI indicator (Relative Strength Index)
    # Calculate the average gain and average loss using Wilder's Smoothing
    # We use a 'com' span of period-1 to match the standard RSI calculation
    delta = df['adjclose'].diff()       # Difference from the previous day
    gain = delta.where(delta > 0, 0)    # Keep gains and replace losses with 0
    loss = -delta.where(delta < 0, 0)   # keep -losses and replace gains with 0
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, adjust=False).mean()  # Average gain
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, adjust=False).mean()  # Average loss
    rs = avg_gain / (avg_loss + 1e-10)  # RS (avoid division by zero)
    df['rsi'] = 100 - (100 / (1 + rs))  # RSI (normalize to a scale of 0 to 100)

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
    
    if plot_type == 'macd':
        macd = df['macd'].tolist()
        signal = df['signal'].tolist()
        histogram = df['histogram'].tolist()
        all_values = macd + signal + histogram
    elif plot_type == 'rsi':
        rsi = df['rsi'].tolist()
        overbought = [RSI_OVERBOUGHT_LEVEL] * len(dates)
        oversold = [RSI_OVERSOLD_LEVEL] * len(dates)
        all_values = rsi + overbought + oversold
    else:
        values = df['adjclose'].tolist()
        ma1 = df['ma1'].tolist()
        ma2 = df['ma2'].tolist()
        ma3 = df['ma3'].tolist()
        all_values = values + ma1 + ma2 + ma3
        
    fig = plotille.Figure()

    # Determine the dimensions and limits of the plot
    fig.width = shutil.get_terminal_size()[0] - 21
    fig.height = math.floor(shutil.get_terminal_size()[1] * height_ratio) - 5
    fig.set_x_limits(dates[0], dates[-1])
    fig.set_y_limits(min(all_values), max(all_values))

    # Prepare the plots and text to display
    if plot_type == 'macd':
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
    else:
        fig.plot(dates, ma3, lc=MOVING_AVG_3_COLOR)
        fig.plot(dates, ma2, lc=MOVING_AVG_2_COLOR)
        fig.plot(dates, ma1, lc=MOVING_AVG_1_COLOR)
        fig.plot(dates, values, lc=PRICE_RATIO_COLOR)
        last = f'{values[-1]:.0f}' if values[-1] > 1000 else f'{values[-1]:.2f}'
        change = f'{(values[-1] / values[0] - 1) * 100:+.0f}'
        text = f'{plot_name} last value: {last} ({change}%)'

    # Display the last value text
    x = dates[0] + (dates[-1] - dates[0]) * 0.55
    y = min(all_values)
    fig.text([x], [y], [text])

    print(fig.show(legend=False))


def log_data_frame(df, description):
    """ This function is used only for debugging. """
    logger.debug(f'DataFrame {description}\n{df}')
    #logger.debug(f'DataFrame index data type: {df.index.dtype}')
    #logger.debug(f'DataFrame index class: {type(df.index)}')
    #logger.debug(f'DataFrame columns data types\n{df.dtypes}')
    #logger.debug(f'DataFrame statistics\n{df.describe()}')  # Mean, min, max, etc.
    sys.exit()


if __name__ == '__main__':
    # Configure the root logger
    logging.basicConfig(level=logging.WARNING,
                        format='%(levelname)s - %(message)s')
    
    # Configure this script's logger
    #logger.setLevel(logging.DEBUG)

    main()
