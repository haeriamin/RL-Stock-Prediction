import os
from finrl import config

import params
import data
import model
import agent


if not os.path.exists('./' + config.DATA_SAVE_DIR):
    os.makedirs('./' + config.DATA_SAVE_DIR)
if not os.path.exists('./' + config.TRAINED_MODEL_DIR):
    os.makedirs('./' + config.TRAINED_MODEL_DIR)
if not os.path.exists('./' + config.TENSORBOARD_LOG_DIR):
    os.makedirs('./' + config.TENSORBOARD_LOG_DIR)
if not os.path.exists('./' + config.RESULTS_DIR):
    os.makedirs('./' + config.RESULTS_DIR)


def main(load_data, load_model):
    if not load_model:
        # Get params
        data_params, env_kwargs, model_params, train_params = params.main()

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
        model_ppo = my_agent.get_model(
            model_name = 'ppo',
            model_kwargs = model_params,
        )
        trained_ppo = my_agent.train(
            model = model_ppo,
            train_kwargs = train_params 
        )
        trained_ppo.save(
            os.path.join(config.TRAINED_MODEL_DIR, 'ppo_model'))
    else:
        pass


if __name__ == '__main__':
    main(load_data=True, load_model=False)








    # # Define A2C agent
    # agent = agent.Agent(
    #     env = env_train
    # )
    # a2c_params = dict(
    #     n_steps = 10,
    #     ent_coef = 0.005,
    #     learning_rate = 0.0004,
    # )
    # a2c_model = agent.get_model(
    #     model_name = 'a2c',
    #     model_kwargs = a2c_params,
    # )
    # trained_a2c = agent.train(
    #     model = a2c_model,
    #     tb_log_name = 'a2c',
    #     total_timesteps = 40000,
    # )
