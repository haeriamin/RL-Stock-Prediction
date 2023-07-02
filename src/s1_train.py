import os
from stable_baselines3 import PPO

import params
import data
import model
import agent


TRAINED_MODEL_DIR = './trained_models'
RESULTS_DIR = './results'
if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def main(load_data, load_model):
        # Get params
        data_params, env_kwargs, model_params, train_params, model_name = params.main()

        # Get training data
        df = data.main(
            stocks = data_params['stocks'],
            mode = 'train',
            load = load_data,
        )

        # Create environment
        train_env = model.StockPortfolioEnv(
            df = df,
            **env_kwargs)
        env_train, _ = train_env.get_sb_env()

        # Define PPO agent
        my_agent = agent.Agent(
            env = env_train,
        )

        if not load_model:
            model_ppo = my_agent.get_model(
                model_name = model_name,
                model_kwargs = model_params,
            )
        else:
            model_ppo = PPO.load(
                os.path.join(TRAINED_MODEL_DIR, model_name),
                env=env_train,
                # **model_params,
            )

        trained_ppo = my_agent.train(
            model = model_ppo,
            train_kwargs = train_params 
        )
        trained_ppo.save(
            os.path.join(TRAINED_MODEL_DIR, model_name))


if __name__ == '__main__':
    main(
        load_data = True,
        load_model = False,
    )
