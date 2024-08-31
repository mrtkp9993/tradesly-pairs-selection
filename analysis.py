import itertools
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint
from tqdm import tqdm

SYMBOLS_FILE = "symbols_bist_100.csv"

logging.basicConfig(format="%(levelname)s:%(name)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
reg = LinearRegression(fit_intercept=True)


def read_stocklist():
    stocks = []
    with open(SYMBOLS_FILE, encoding="utf-8") as f:
        for line in f:
            stocks.append({"symbol": line.strip()})
    return stocks


def download_data(stocks):
    data_dict = {}
    for stock in tqdm(stocks):
        symbol = stock["symbol"]
        data = yf.download(symbol, period="2y", interval="1d", progress=False)
        if not data.empty and len(data) > 252:
            data_dict[symbol] = data
    return data_dict


def main():
    logger.info(f"{__name__} started")
    logger.info(f"Reading stocklist from {SYMBOLS_FILE}")
    stocks = read_stocklist()
    logger.info(f"Downloading data for {len(stocks)} stocks")
    data = download_data(stocks)

    logger.info("Generating pairs list")
    pairs = list(itertools.combinations(data.keys(), 2))
    logger.info(f"Found {len(pairs)} pairs")

    logger.info("Calculating cointegration")
    cointegrated_pairs = []
    for pair in tqdm(pairs):
        stock1 = data[pair[0]]["Adj Close"]
        stock2 = data[pair[1]]["Adj Close"]
        # Align the series by their dates
        combined_data = pd.concat([stock1, stock2], axis=1, join="inner")
        combined_data.columns = [pair[0], pair[1]]

        _, p_value, _ = coint(combined_data[pair[0]], combined_data[pair[1]])
        if p_value < 0.05:
            X = sm.add_constant(combined_data[pair[0]])
            y = combined_data[pair[1]]
            model = sm.OLS(y, X).fit()
            beta = model.params[pair[0]]
            spread = combined_data[pair[1]] - beta * combined_data[pair[0]]
            cointegrated_pairs.append(
                {
                    "pair": pair,
                    "spread": spread,
                }
            )
    logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")

    logger.info("Calculating Hurst exponent for each spread")
    hurst_pairs = []
    for pair in tqdm(cointegrated_pairs):
        spread = pair["spread"].to_numpy()

        lags = range(2, 100)
        tau = [
            np.sqrt(np.std(np.subtract(spread[lag:], spread[:-lag]))) for lag in lags
        ]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        hurst = poly[0] * 2.0

        if hurst < 0.5:
            hurst_pairs.append(pair)
    logger.info(f"Found {len(hurst_pairs)} pairs with Hurst exponent < 0.5")

    logger.info("Calculating half-life for each spread")
    half_life_pairs = []
    for pair in tqdm(hurst_pairs):
        spread = pair["spread"]

        X = spread.shift(1).dropna().to_numpy().reshape(-1, 1)
        y = spread.diff().dropna().to_numpy()

        reg.fit(X, y)

        half_life = -np.log(2) / reg.coef_[0]

        if half_life < 252 and half_life > 1:
            half_life_pairs.append(pair)
    logger.info(
        f"Found {len(half_life_pairs)} pairs with half-life between 1 and 252 days"
    )

    logger.info("Calculating mean cross for each spread")
    mean_cross_pairs = []
    for pair in tqdm(half_life_pairs):
        spread = pair["spread"]
        centered_spread = spread - spread.mean()
        cross_over_indices = np.where(np.diff(np.sign(centered_spread)))[0]
        if len(cross_over_indices) > 12:
            mean_cross_pairs.append(pair)

    logger.info(f"Found {len(mean_cross_pairs)} pairs with mean cross > 12")

    # write pairs list to file with buy / sell signals
    with open("pairs.txt", "w") as f:
        for pair in mean_cross_pairs:
            f.write(f"{pair['pair'][0]},{pair['pair'][1]}\n")
    logger.info("Pairs list written to pairs.txt")


if __name__ == "__main__":
    main()
