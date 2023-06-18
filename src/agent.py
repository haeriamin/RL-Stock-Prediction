# DRL models from Stable Baselines 3
import numpy as np
from finrl import config
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


MODELS = {
    "a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO
}
MODEL_KWARGS = {
    x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()
}
NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException:
            self.logger.record(key="train/reward", value=self.locals["reward"][0])
        return True


class Agent:
    """Provides implementations for DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """
    def __init__(self, env):
        self.env = env


    def get_model(self, model_name, model_kwargs=None):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
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
    def predict(model_name, environment, cwd, deterministic=True):
        # load agent
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        try:
            model = MODELS[model_name].load(cwd)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # make a prediction
        test_env, test_obs = environment.get_sb_env()
        test_env.reset()

        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = test_env.step(action)
            
            if i == (len(environment.df.index.unique()) - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            
            if dones[0]:
                break

        return account_memory[0], actions_memory[0]
