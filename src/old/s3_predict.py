import os
import warnings
import pandas as pd
from finrl import config
import pandas_market_calendars as mcal


import params
import data
import model
import agent


warnings.filterwarnings("ignore")


def main(data_params, env_kwargs, model_name, date):
    # Get data
    test = data.main(
        stocks = data_params['stocks'],
        mode = 'predict',
        load = False,
        date = date,
    )

    # Create environment
    test_env = model.StockPortfolioEnv(
        df = test,
        **env_kwargs
    )

    # Predict
    df_daily_return_ppo, df_actions_ppo = agent.Agent.predict(
        model_name = model_name,
        environment = test_env,
        cwd = os.path.join(config.TRAINED_MODEL_DIR, model_name),
        deterministic = True,
    )
    print('\n*** Actions ***\n', df_actions_ppo)

    # Account value calculation
    ppo_cumprod = (df_daily_return_ppo.daily_return + 1).cumprod() - 1
    account_value = env_kwargs['initial_amount'] + (env_kwargs['initial_amount'] * ppo_cumprod)
    print('\n*** Account value ***\n', account_value)

    return account_value.iat[-1], df_actions_ppo.iloc[-1, :].values.flatten().tolist()


if __name__ == '__main__':
    # Get params
    data_params, env_params, model_params, _ = params.main()

    # Last business dates
    nyse = mcal.get_calendar('NYSE')

    end_date = pd.to_datetime('today')
    start_date = end_date - pd.to_timedelta(30, unit='d')
    days = nyse.schedule(start_date=start_date, end_date=end_date)

    length = 4
    start_date = str(days.index[-length])[:10]
    end_date = str(end_date)[:10]
    date = [start_date, end_date]

    # Params
    env_params['initial_amount'] = 10
    env_params['initial_allocation'] = [0.6, 0.4]

    # Run
    initial_amount, initial_allocation = main(
        data_params,
        env_params,
        model_params['model_name'],
        date,
    )
