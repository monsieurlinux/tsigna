![NVDA stock price, moving averages and volume over 1 year](https://github.com/monsieurlinux/tsigna/raw/main/img/tsigna-nvda-moving-averages-volume-1y.png "NVDA stock price, moving averages and volume over 1 year")

# Tsigna

[![PyPI][pypi-badge]][pypi-link]
[![License][license-badge]][license-link]

Tsigna is a Python financial analysis tool that runs entirely in the terminal. Designed for medium-term trading, it fetches historical stock data from Yahoo Finance to generate text-based charts using the plotille library.

Tsigna calculates technical indicators like moving averages, MACD, and RSI, while offering unique features such as **ratio comparison** between two tickers in addition to single-ticker analysis. Users can customize the timeframe and view multiple indicators simultaneously.

![NVDA stock price, Bollinger Bands and Stochastics over 1 year](https://github.com/monsieurlinux/tsigna/raw/main/img/tsigna-nvda-bollinger-bands-stochastics-1y.png "NVDA stock price, Bollinger Bands and Stochastics over 1 year")

## Background

I originally looked for a free online tool to plot the **ratio between two tickers** but couldn't find one, so I started working on Tsigna. The name combines 'T' for terminal and the plural of *signum*, the Latin word for signal. The 'T' also stands for technical, as the tool is designed to generate technical signals.

![Ratio between NVDA and WMT stock prices, moving averages and RSI indicator over 2 years](https://github.com/monsieurlinux/tsigna/raw/main/img/tsigna-nvda-wmt-moving-averages-rsi-2y.png "Ratio between NVDA and WMT stock prices, moving averages and RSI indicator over 2 years")

## Dependencies

Tsigna requires the following external libraries:

* **[pandas][pandas-link]**: Used for data manipulation and analysis.
* **[plotille][plotille-link]**: Used for creating terminal-based plots.
* **[yahooquery][yahooquery-link]**: Used for fetching financial data from Yahoo Finance.

These libraries and their sub-dependencies will be installed automatically when you install Tsigna.

## Installation

It is recommended to install Tsigna within a [virtual environment][venv-link] to avoid conflicts with system packages. Some Linux distributions enforce this. You can use `pipx` to handle the virtual environment automatically, or create one manually and use `pip`.

### Installation with `pipx`

`pipx` installs Tsigna in an isolated environment and makes it available globally.

**1. Install `pipx`:**

*   **Linux (Debian / Ubuntu / Mint):**
    
    ```bash
    sudo apt install pipx
    pipx ensurepath
    ```
*   **Linux (Other) / macOS:**
    
    ```bash
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    ```
*   **Windows:**
    
    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

You may need to reopen your terminal for the PATH changes to take effect. If you encounter a problem, please refer to the official [pipx documentation][pipx-link].

**2. Install Tsigna:**

```bash
pipx install tsigna
```

### Installation with `pip`

If you prefer to manage the virtual environment manually, you can create and activate it by following this [tutorial][venv-link]. Then install Tsigna:

```bash
pip install tsigna
```

## Deployments

View all releases on:

- **[PyPI Releases][pypi-releases]**
- **[GitHub Releases][github-releases]**

## Usage

### Basic Usage

```bash
tsigna [arguments] [TICKER1] [TICKER2]
```

### Arguments

| Argument           | Short Flag | Description                                                  |
| ------------------ | ---------- | ------------------------------------------------------------ |
| `--help`           | `-h`       | Show help message                                            |
| `--atr`            | `-a`       | Display ATR indicator (Average True Range)                   |
| `--atr-only`       | `-A`       | Display **only** ATR indicator                               |
| `--bollinger`      | `-b`       | Display Bollinger Bands indicator                            |
| `--mfi`            | `-f`       | Display MFI indicator (Money Flow Index)                     |
| `--mfi-only`       | `-F`       | Display **only** MFI indicator                               |
| `--indicator-info` | `-i`       | Show indicator information                                   |
| `--log-scale`      | `-l`       | Use a logarithmic scale on the y-axis                        |
| `--macd`           | `-m`       | Display MACD indicator (Moving Average Convergence Divergence) |
| `--macd-only`      | `-M`       | Display **only** MACD indicator                              |
| `--no-cache`       | `-n`       | Bypass cache and get latest data                             |
| `--obv`            | `-o`       | Display OBV indicator (On-Balance Volume)                    |
| `--obv-only`       | `-O`       | Display **only** OBV indicator                               |
| `--rsi`            | `-r`       | Display RSI indicator (Relative Strength Index)              |
| `--rsi-only`       | `-R`       | Display **only** RSI indicator                               |
| `--stoch`          | `-s`       | Display Stochastics indicator                                |
| `--stoch-only`     | `-S`       | Display **only** Stochastics indicator                       |
| `--version`        | `-v`       | Show program's version number and exit                       |
| `--volume`         | `-w`       | Display volume                                               |
| `--volume-only`    | `-W`       | Display **only** volume                                      |
| `--years`          | `-y`       | Set years to plot, use 0 for ytd (default: 1)                |

## Configuration

When you run Tsigna for the first time, a `config.toml` file is automatically created. Its location depends on your operating system (typical paths are listed below):

*   **Linux:** `~/.config/tsigna`
*   **macOS:** `~/Library/Preferences/tsigna`
*   **Windows:** `C:/Users/YourUsername/AppData/Roaming/tsigna`

You can edit this file to customize various settings. Common customizations include changing the expiration time of the cache (or disabling it), modifying the line colors, or changing the parameters of the technical indicators.

## Not Financial Advice

I do not own any of the stocks in the examples, I chose them because they are very popular.

## License

Copyright (c) 2025 Monsieur Linux

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

Tsigna is not doing much more than getting data from [yahooquery][yahooquery-link], processing it with [pandas][pandas-link], and plotting it with [plotille][plotille-link], so thanks to the creators and contributors of these powerful libraries for making it possible.

Thanks also to the [ticker][ticker-link] tool, which is very useful to track prices in real time from the terminal.

[github-releases]: https://github.com/monsieurlinux/tsigna/releases
[license-badge]: https://img.shields.io/pypi/l/tsigna.svg
[license-link]: https://github.com/monsieurlinux/tsigna/blob/main/LICENSE
[pandas-link]: https://github.com/pandas-dev/pandas
[pipx-link]: https://github.com/pypa/pipx
[plotille-link]: https://github.com/tammoippen/plotille
[pypi-badge]: https://img.shields.io/pypi/v/tsigna.svg
[pypi-link]: https://pypi.org/project/tsigna/
[pypi-releases]: https://pypi.org/project/tsigna/#history
[ticker-link]: https://github.com/achannarasappa/ticker
[venv-link]: https://docs.python.org/3/tutorial/venv.html
[yahooquery-link]: https://github.com/dpguthrie/yahooquery

