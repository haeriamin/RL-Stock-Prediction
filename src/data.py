import os
import warnings
import pandas as pd
from finrl import config, config_tickers

import downloader
import preprocessor


warnings.filterwarnings("ignore")


def main(stocks, mode, load, date=None):
    if not load:
        df = downloader.YahooDownloader(
            start_date = '2017-01-01', #'2008-01-01',
            end_date = '2023-06-01',
            ticker_list = stocks, #config_tickers.DOW_30_TICKER,
        ).fetch_data()

        df.to_csv(
            os.path.join('./', config.DATA_SAVE_DIR, 'raw_nsdq7.csv'),
            index = False)
    else:
        df = pd.read_csv(
            os.path.join('./', config.DATA_SAVE_DIR, 'raw_nsdq7.csv'))
    
    fe = preprocessor.FeatureEngineer(
        use_technical_indicator = True,
        use_turbulence = False,
        user_defined_feature = False,
    )

    df = fe.preprocess_data(df)

    # Add covariance matrix as states
    df = df.sort_values(['date', 'tic'], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # Look back is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback : i, :]
        price_lookback = data_lookback.pivot_table(
            index='date', columns='tic', values='close'
        )
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)

    df_cov = pd.DataFrame(
        {
            'date': df.date.unique()[lookback:],
            'cov_list': cov_list,
            'return_list': return_list,
        }
    )
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)

    if mode == 'train':
        df = preprocessor.data_split(df, '2017-01-01', '2023-01-01')
    elif mode == 'test':
        df = preprocessor.data_split(df, date[0], date[1])

    return df
