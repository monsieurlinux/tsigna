![NVDA stock price, moving averages and volume over 1 year](https://github.com/monsieurlinux/tsigna/raw/main/img/tsigna-nvda-moving-averages-volume-1y.png "NVDA stock price, moving averages and volume over 1 year")

# Tsigna

Tsigna is a Python financial analysis tool that runs entirely in the terminal. It is most useful for medium-term trading. It fetches historical stock data from Yahoo Finance, calculates technical indicators including moving averages, MACD, and RSI, and displays them as text-based charts using the plotille library. The tool supports single ticker analysis, ratio comparisons between two tickers, and a special MMRI calculation. Users can customize the time period and split the terminal display to show multiple indicators simultaneously.

![NVDA stock price, Bollinger Bands and Stochastics over 1 year](https://github.com/monsieurlinux/tsigna/raw/main/img/tsigna-nvda-bollinger-bands-stochastics-1y.png "NVDA stock price, Bollinger Bands and Stochastics over 1 year")

## Background

Originally I was looking for a free online tool to plot the **ratio between two tickers**, but I didn't find such a tool so I started working on Tsigna. The name comes from 'T' for terminal and the plural form of signum, the latin word for signal. The 'T' also stands for technical, like in technical indicators, from which we get technical signals.

![Ratio between NVDA and WMT stock prices, moving averages and RSI indicator over 2 years](https://github.com/monsieurlinux/tsigna/raw/main/img/tsigna-nvda-wmt-moving-averages-rsi-2y.png "Ratio between NVDA and WMT stock prices, moving averages and RSI indicator over 2 years")

## Installation

Tsigna has been developped with Python 3.11 but may work with older versions. It depends on the [pandas](https://github.com/pandas-dev/pandas), [plotille](https://github.com/tammoippen/plotille) and [yahooquery](https://github.com/dpguthrie/yahooquery) external libraries and their dependencies. They will all be installed automatically with the following command. It is recommended to make the installation within a [virtual environment](https://docs.python.org/3/tutorial/venv.html).

```bash
pip install tsigna
```

## Usage

### Basic Usage

```bash
tsigna [arguments] ticker1 [ticker2]
```

### Command-Line Arguments

| Argument           | Short Flag | Description                                        |
| ------------------ | ---------- | -------------------------------------------------- |
| `--help`           | `-h`       | Show help message                                  |
| `--atr`            | `-a`       | Display ATR indicator (Average True Range)         |
| `--atr-only`       | `-A`       | Display **only** ATR indicator                     |
| `--bollinger`      | `-b`       | Display Bollinger Bands indicator                  |
| `--mfi`            | `-f`       | Display MFI indicator (Money Flow Index)           |
| `--mfi-only`       | `-F`       | Display **only** MFI indicator                     |
| `--indicator-info` | `-i`       | Show indicator information                         |
| `--macd`           | `-m`       | Display MACD indicator (Moving Average Convergence Divergence) |
| `--macd-only`      | `-M`       | Display **only** MACD indicator                    |
| `--no-cache`       | `-n`       | Bypass cache and get latest data                   |
| `--obv`            | `-o`       | Display OBV indicator (On-Balance Volume)          |
| `--obv-only`       | `-O`       | Display **only** OBV indicator                     |
| `--rsi`            | `-r`       | Display RSI indicator (Relative Strength Index)    |
| `--rsi-only`       | `-R`       | Display **only** RSI indicator                     |
| `--stoch`          | `-s`       | Display Stochastics indicator                      |
| `--stoch-only`     | `-S`       | Display **only** Stochastics indicator             |
| `--volume`         | `-v`       | Display volume                                     |
| `--volume-only`    | `-V`       | Display **only** volume                            |
| `--years`          | `-y`       | Set years to plot, use 0 for ytd (default: 1)      |

## Configuration

You can edit the configuration constants directly at the top of the tsigna.py file if you wish to change the default behavior. For example you can change the expiration time of the cache (it is 5 minutes by default) or disable it. You can also change the colors of the lines, the parameters of the technical indicators, etc.

## Not Financial Advice

I do not own any of the stocks in the examples, I chose them because they are very popular.

## License

Copyright (c) 2025 Monsieur Linux

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Tsigna is not doing much more than getting data from [yahooquery](https://github.com/dpguthrie/yahooquery), processing it with [pandas](https://github.com/pandas-dev/pandas), and plotting it with [plotille](https://github.com/tammoippen/plotille), so thanks to the creators and contributors of these powerful libraries for making it possible.

Thanks also to the [ticker](https://github.com/achannarasappa/ticker) tool, which is very useful to track prices in real time from the terminal.
