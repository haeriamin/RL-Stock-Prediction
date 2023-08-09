import os
import pickle
import warnings
import pandas as pd

import downloader
import preprocessor


warnings.filterwarnings('ignore')

DATASETS_DIR = './datasets/'
if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR)


def main(data_params, mode, load, date=None):
    if not load:
        df = downloader.YahooDownloader(
            start_date = '2017-01-01',
            end_date = '2023-06-01',
            ticker_list = data_params['stocks'],
        ).fetch_data()

        df.to_csv(
            # os.path.join(DATASETS_DIR, 'raw_nsdq6.csv'),
            os.path.join(DATASETS_DIR, 'raw_dj7.csv'),
            index = False)
    else:
        df = pd.read_csv(
            # os.path.join(DATASETS_DIR, 'raw_nsdq6.csv'))
            os.path.join(DATASETS_DIR, 'raw_dj7.csv'))
    
    fe = preprocessor.FeatureEngineer(
        use_technical_indicator = True,
        use_turbulence = False,
        user_defined_feature = False,
    )

    df = fe.preprocess_data(df)

    # # Add covariance matrix
    # df = df.sort_values(['date', 'tic'], ignore_index=True)
    # df.index = df.date.factorize()[0]
    # cov_list = []
    # return_list = []
    # # Look back is one year
    # lookback = 252
    # for i in range(lookback, len(df.index.unique())):
    #     data_lookback = df.loc[i - lookback : i, :]
    #     price_lookback = data_lookback.pivot_table(
    #         index='date', columns='tic', values='close'
    #     )
    #     return_lookback = price_lookback.pct_change().dropna()
    #     return_list.append(return_lookback)

    #     covs = return_lookback.cov().values
    #     cov_list.append(covs)

    # df_cov = pd.DataFrame(
    #     {
    #         'date': df.date.unique()[lookback:],
    #         'cov_list': cov_list,
    #         'return_list': return_list,
    #     }
    # )
    # df = df.merge(df_cov, on='date')

    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    df.drop(labels='open', axis=1, inplace=True)
    df.drop(labels='high', axis=1, inplace=True)
    df.drop(labels='low', axis=1, inplace=True)
    df['close_org'] = df['close'].copy()

    if mode == 'train':
        df = preprocessor.data_split(df, '2017-01-01', '2023-01-01')

        # Normalization
        features_mean = df[data_params['feature_list']].mean()
        features_std = df[data_params['feature_list']].std()
        with open(os.path.join(DATASETS_DIR, 'fit_features_means_and_stds.pickle'), 'wb') as f:
            pickle.dump([features_mean, features_std], f)
        # df.to_csv(DATASETS_DIR + 'sample.csv', index=False)

        df[data_params['feature_list']] = (df[data_params['feature_list']] - features_mean) / features_std

    elif mode == 'test':
        df = preprocessor.data_split(df, date[0], date[1])

        # Normalization
        with open(os.path.join(
            DATASETS_DIR, 'fit_features_means_and_stds.pickle'), 'rb') as f:
            features_mean, features_std = pickle.load(f)
        
        df[data_params['feature_list']] = (df[data_params['feature_list']] - features_mean) / features_std

    return df
