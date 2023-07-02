import policy


def main():
    model_name = 'ppo' # 'lstmppo'
    history_window = 5

    data_params = dict(
        stocks = [
            'AAPL',
            # 'MSFT',
            # 'GOOG',
            # 'AMZN',
            'TSLA',
            # 'NVDA',
        ],  # TODO: Add META/FB
        
        features_list = [
            'close',
            'volume',
        ],


        tech_indicator_list = [
            'macd',
            'boll_ub', 'boll_lb',
            'rsi_30', 'cci_30', 'dx_30',
            'close_30_sma', 'close_60_sma',
        ],

        test_dates = [
            # ['2023-01-03', '2023-01-05'], # i.e. on 3rd only (4th is to measure reward)

            # ['2023-01-01', '2023-02-01'],
            # ['2023-02-01', '2023-03-01'],
            # ['2023-03-01', '2023-04-01'],
            # ['2023-04-01', '2023-05-01'],
            # ['2023-05-01', '2023-06-01'],

            ['2023-01-01', '2023-06-01'],
        ],
    )

    # Environment parameters
    env_params = dict(
        initial_amount = 1000,
        commission_perc = 1,  # transaction cost percentage per trade [%]
        reward_scaling = 1,  # scaling factor for reward, good for training
        initial_allocation = [0.7, 0.3],#[1 / len(data_params['stocks'])] * len(data_params['stocks']),
        # day = # an increment number to control date
        history_window = history_window,

        state_space = len(data_params['stocks']),  # dimension of input features
        stock_dim = len(data_params['stocks']),  # number of unique stocks
        action_space = len(data_params['stocks']),  # equals stock_dim
        turbulence_threshold = len(data_params['stocks']), # equals stock_dim
        tech_indicator_list = data_params['tech_indicator_list'],  # list of technical indicator names
    )

    def linear_schedule(initial_value):
        def func(progress_remaining):
            # Progress will decrease from 1 (beginning) to 0.
            return progress_remaining * initial_value
        return func

    model_params = dict(
        # env =  # Will be passed elsewhere
        policy = policy.ActorCriticPolicy  # The policy model (MlpPolicy, CnnPolicy, MultiInputPolicy)

        learning_rate = linear_schedule(0.001), # The learning rate, it can be a function of the current progress remaining (from 1 to 0) | Def: 3e-4
        batch_size = 2 ** 6,  # Def: 2 ** 6
        n_epochs = 10,  # Number of epoch when optimizing the surrogate loss | Def: 10
        gamma = 0.99,  # Discount factor | Def: 0.99
        gae_lambda = 0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range = 0.2,  # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
        normalize_advantage = True,  # Whether to normalize or not the advantage
        ent_coef = 0.001,  # Entropy coefficient for the loss calculation | Def: 0.0 | Increase agent's exploration
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
        n_steps = 2 ** 11,  # Def: 2 ** 11

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
        target_kl = 0.015,  # Def: None 
    )
    
    train_params = dict(
        tb_log_name = 'ppo',
        total_timesteps = 2 ** 18,  
        log_interval = 1,
        reset_num_timesteps = True,
        progress_bar = True,
    )

    print('Stocks:', data_params['stocks'])
    return data_params, env_params, model_params, train_params, model_name


    # Notes:
    # num_envs = batch_size / n_steps
    # minibatch_size = batch_size / num_envs
    # num_updates = total_timesteps / batch_size
    
    # lr = (1 - (update - 1) / num_updates) * initial_lr
    
    # loss = policy_gradient_loss (advantage/actor) + value_loss (reward) - entropy_loss (exploration)

    # P(s', r|s, a) : Probability of getting state s' and reward r given s and action a
    # E(s_t) = SIG r SIG P(s', r|s, a) : Expected values is equal to probabilities multiplied by rewards

    # G_t = r_(t+1) + gamma G_(t+1) : return

    # v_pi(s) = E_pi[G_t|S_t=s] = E_pi[SIG gamma^k R_(t+k+1)|S_t=s] : value function
    # q_pi(s, a) = E_pi[G_t|S_t=s, A_t=a] = E_pi[SIG gamma^k R_(t+k+1)|S_t=s, A_t=a] : action-value (quality) function

    # Actor-Critic Method:
    #   - Actor approximates policy | Policy(state) -> action
    #   - Critic approximates value function, v(s) | value(state, action) -> reward
        
    # explained_var: Tells if the value function is a good indicator of the returns (rewards)
    
    # 1/ The loss function, as you have said, is not a proxy for how well the agent is doing. This is partly due to the non-stationary nature of RL. For example in a game moving to a new room in the game could be seen as progress, but would negatively impact the loss function as everything is novel.
    # 2/ A useful metric (from spinning up as far as I remember) is to measure the explained variance between the predicted value and actual returns for the agent. This seems to be a very helpful indicator of whether the agent is learning the problem or not.
    # 3/ Sometimes other metrics can be useful proxies for progress. For example, the episode length can sometimes be a better indicator of progress than the actual score.

