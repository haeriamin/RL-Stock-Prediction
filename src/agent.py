import numpy as np
# from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO #, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


MODELS = {
    'ppo': PPO,
    # 'lstmppo': RecurrentPPO,
    # 'a2c': A2C,
    # 'ddpg': DDPG,
    # 'td3': TD3,
    # 'sac': SAC,
}

NOISE = {
    'normal': NormalActionNoise,
    'ornstein_uhlenbeck': OrnsteinUhlenbeckActionNoise,
}


class TensorboardCallback(BaseCallback):
    # Custom callback for plotting additional values in tensorboard.
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key='train/reward', value=self.locals['rewards'][0])
        except BaseException:
            self.logger.record(key='train/reward', value=self.locals['reward'][0])
        return True


class Agent:
    def __init__(self, env):
        self.env = env


    def get_model(self, model_name, model_kwargs=None):
        if model_name not in MODELS:
            raise NotImplementedError('NotImplementedError')

        if 'action_noise' in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs['action_noise'] = NOISE[model_kwargs['action_noise']](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        return MODELS[model_name](
            env = self.env,
            **model_kwargs,
        )


    def train(self, model, train_kwargs=None):
        model = model.learn(
            callback = TensorboardCallback(),
            **train_kwargs,
        )
        return model


    @staticmethod
    def predict(model_name, environment, cwd, history_window, deterministic=True):
        # load agent
        model = MODELS[model_name].load(cwd)

        # make a prediction
        env, observation = environment.get_sb_env()
        # observation = env.reset()
        episodic_reward = 0

        for i in range(len(environment.df.index.unique())):
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, done, info = env.step(action)

            episodic_reward += reward

            if i == (len(environment.df.index.unique()) - 2 - history_window):
                account_memory = env.env_method(method_name='save_asset_memory')
                actions_memory = env.env_method(method_name='save_action_memory')
            
            if done[0]:
                print(episodic_reward)
                episodic_reward = 0
                break

        return account_memory[0], actions_memory[0]
