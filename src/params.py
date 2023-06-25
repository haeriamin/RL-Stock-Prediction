def main():
    data_params = dict(
        stocks = [
            'AAPL',
            # 'MSFT',
            # 'GOOG',
            # 'AMZN',
            'TSLA',
            # 'NVDA',
        ],  # TODO: Add META/FB
        
        tech_indicator_list = [
            "macd",
            "boll_ub", "boll_lb",
            "rsi_30", "cci_30", "dx_30",
            "close_30_sma", "close_60_sma",
        ],

        test_dates = [
            ['2023-01-01', '2023-02-01'],
            ['2023-02-01', '2023-03-01'],
            ['2023-03-01', '2023-04-01'],
            ['2023-04-01', '2023-05-01'],
            ['2023-05-01', '2023-06-01'],
        ],
    )

    # Environment parameters
    env_params = dict(
        initial_amount = 1000,
        commission_perc = 2,  # TODO: transaction cost percentage per trade [%]
        reward_scaling = 1,  # scaling factor for reward, good for training
        initial_allocation = [1 / len(data_params['stocks'])] * len(data_params['stocks']),
        # day = # an increment number to control date

        state_space = len(data_params['stocks']),  # dimension of input features
        stock_dim = len(data_params['stocks']),  # number of unique stocks
        action_space = len(data_params['stocks']),  # equals stock_dim
        turbulence_threshold = len(data_params['stocks']), # equals stock_dim
        tech_indicator_list = data_params['tech_indicator_list'],  # list of technical indicator names
    )

    model_params = dict(
        # env =  # Will be passed elsewhere

        # model_name = 'ppo', #'ppo',
        # policy = 'MlpPolicy', #'MlpPolicy',  # The policy model to use (MlpPolicy, CnnPolicy, MultiInputPolicy)

        model_name = 'lstmppo',
        policy = 'MlpLstmPolicy',

        learning_rate = 3e-4,  # The learning rate, it can be a function of the current progress remaining (from 1 to 0) | Def: 3e-4
        batch_size = 2 ** 12,  # Def: 2 ** 6
        n_epochs = 50,  # Number of epoch when optimizing the surrogate loss | Def: 10
        gamma = 0.99,  # Discount factor | Def: 0.99
        gae_lambda = 0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range = 0.2,  # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
        normalize_advantage = True,  # Whether to normalize or not the advantage
        ent_coef = 0,#0.005,  # Entropy coefficient for the loss calculation | Def: 0.0
        vf_coef = 0.5,  # Value function coefficient for the loss calculation
        max_grad_norm = 0.5,  # The maximum value for the gradient clipping
        use_sde = False,  # Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
        sde_sample_freq = -1,  # Sample a new noise matrix every n steps when using gSDE | Def: -1 (only sample at the beginning of the rollout)

        tensorboard_log = './tensorboard_log/',  # the log location for tensorboard (if None, no logging)
        policy_kwargs = None,  # Additional arguments to be passed to the policy on creation
        verbose = 0, # Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
        seed = 42,  # Seed for the pseudo random generators
        device = 'auto',  # Device (cpu, cuda, ...) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
        _init_setup_model = True,  # Whether or not to build the network at the creation of the instance | Def: True

        # The number of steps to run for each environment per update
        # (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        # NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        # See https://github.com/pytorch/pytorch/issues/29372
        n_steps = 2 ** 12,  # Def: 2 ** 11

        # Clipping parameter for the value function,
        # it can be a function of the current progress remaining (from 1 to 0).
        # This is a parameter specific to the OpenAI implementation. If None is passed (default),
        # no clipping will be done on the value function.
        # IMPORTANT: this clipping depends on the reward scaling.
        clip_range_vf = None,  # Def: None

        # Limit the KL divergence between updates,
        # because the clipping is not enough to prevent large update
        # see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        # By default, there is no limit on the kl div.
        target_kl = None,  # Def: None
    )
    
    train_params = dict(
        tb_log_name = 'PPO',
        total_timesteps = 80e4,
        log_interval = 1,
        reset_num_timesteps = True,
        progress_bar = True,
    )

    print('Stocks:', data_params['stocks'])
    return data_params, env_params, model_params, train_params
