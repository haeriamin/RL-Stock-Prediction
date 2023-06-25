import os
import tkinter
import warnings
# import pandas as pd
import matplotlib as mpl
from finrl import config
# import plotly.express as px
import matplotlib.pyplot as plt
from pyfolio import timeseries, plotting
from finrl.plot import \
    backtest_stats, get_daily_return, \
    get_baseline, convert_daily_return_to_pyfolio_ts

# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns
# from pypfopt import objective_functions

import params
import data
import model
import agent


mpl.use('TkAgg')
warnings.filterwarnings("ignore")


def main(data_params, env_kwargs, model_name, date, load_data):
    # Get data
    test = data.main(
        stocks = data_params['stocks'],
        mode = 'test',
        load = load_data,
        date = date,
    )

    # Create environment
    test_env = model.StockPortfolioEnv(
        df = test,
        **env_kwargs
    )

    # Info
    unique_trade_date = test.date.unique()
    print('\nDate range:', unique_trade_date[0], unique_trade_date[-1])

    # # TODO: Baseline (DJIA)
    # baseline_df = get_baseline(
    #     ticker = '^DJI',
    #     start = '2020-07-01',
    #     end = '2021-09-01'
    # )
    # baseline_df_stats = backtest_stats(baseline_df, value_col_name='close')
    # baseline_returns = get_daily_return(baseline_df, value_col_name='close')
    # dji_cumprod = (baseline_returns + 1).cumprod() - 1
    # print('*** DJI cum prod ***\n', dji_cumprod)

    # Predict
    df_daily_return_ppo, df_actions_ppo = agent.Agent.predict(
        model_name = model_name,
        environment = test_env,
        cwd = os.path.join(config.TRAINED_MODEL_DIR, model_name),
        deterministic = True,
    )
    # print('\n*** df_daily_return_ppo ***\n', df_daily_return_ppo)
    print('\n*** Actions ***\n', df_actions_ppo)
    ppo_cumprod = (df_daily_return_ppo.daily_return + 1).cumprod() - 1

    # Pyfolio backtest
    strat_ppo = convert_daily_return_to_pyfolio_ts(df_daily_return_ppo)
    perf_stats_all_ppo = timeseries.perf_stats(
        returns = strat_ppo,
        factor_returns = strat_ppo,
        positions = None,
        transactions = None,
        turnover_denom = 'AGB',
    )
    # print('\n*** Performance stats ***\n', perf_stats_all_ppo)

    # fig, ax = plt.subplots()
    # ax.set_title('test title')
    # ax = plotting.plot_perf_stats(
    #     returns = strat_ppo,
    #     factor_returns = strat_ppo,
    #     ax = ax)
    # plt.show()

    # Account value calculation
    account_value = env_kwargs['initial_amount'] + (env_kwargs['initial_amount'] * ppo_cumprod)
    print('\n*** Account value ***\n', account_value)
    plt.plot(df_daily_return_ppo.date, account_value)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(
        config.RESULTS_DIR, 'account_value'),
        dpi=400
    )

    return account_value.iat[-1], df_actions_ppo.iloc[-1, :].values.flatten().tolist()


if __name__ == '__main__':
    # Get params
    data_params, env_params, model_params, _ = params.main()

    # Run
    for date in data_params['test_dates']:
        initial_amount, initial_allocation = main(data_params, env_params, model_params['model_name'], date, load_data=True)
        env_params['initial_amount'] = initial_amount
        env_params['initial_allocation'] = initial_allocation
