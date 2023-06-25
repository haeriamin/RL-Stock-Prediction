import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


matplotlib.use("Agg")


class StockPortfolioEnv(gym.Env):
    """A portfolio allocation environment for OpenAI gym
    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            equals stock dimension
        day: int
            an increment number to control date
    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        initial_amount,
        initial_allocation,
        commission_perc,
        state_space,
        action_space,
        tech_indicator_list,
        reward_scaling = 1,
        turbulence_threshold=None,
        lookback=252,
        day=0,
    ):
        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.initial_allocation = initial_allocation
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.commission_perc = commission_perc
        self.reward_scaling = reward_scaling

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(
            low = 0,
            high = 1,
            shape = (self.action_space,)
        )
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (
                self.state_space + len(self.tech_indicator_list),
                self.state_space,
            ),
        )

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0)
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount

        # memorize values each step
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [self.initial_allocation]
        self.date_memory = [self.data.date.unique()[0]]


    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ["daily_return"]

            plt.plot(df.daily_return.cumsum(), "r")
            plt.savefig("results/cumulative_reward.png")
            plt.close()
            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig("results/rewards.png")
            plt.close()

            # print("begin_total_asset:{}".format(self.asset_memory[0]))
            # print("end_total_asset:{}".format(self.portfolio_value))

            # Calculate sharpe ratio
            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                # print("Sharpe ratio: ", sharpe)

            return self.state, self.reward, self.terminal, {}

        else:
            allocation = self.softmax_normalization(actions)
            self.actions_memory.append(allocation)
            last_day_memory = self.data

            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.covs = self.data["cov_list"].values[0]
            self.state = np.append(
                np.array(self.covs),
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                axis=0,
            )

            # Ratio of portfolio return (in [-1, 1])
            return_ratio = sum(
                ((self.data.close.values / last_day_memory.close.values) - 1) * allocation)

            # Calculate commission fee
            commission_fee = self.commission_perc / 100 * self.portfolio_value * \
                np.sum(np.abs(allocation - self.actions_memory[-2]))
            commission_ratio = commission_fee / self.portfolio_value

            # New portfolio value
            self.portfolio_value = self.portfolio_value * (1 + return_ratio) - commission_fee

            # Reward [%]
            self.reward = self.reward_scaling * (
                return_ratio -
                commission_ratio
            ) * 100
            # print(commission_fee, commission_ratio * 100)

            # Save into memory
            self.portfolio_return_memory.append(return_ratio)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(self.portfolio_value)

        return self.state, self.reward, self.terminal, {}


    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]

        self.asset_memory = [self.initial_amount]
        self.actions_memory = [self.initial_allocation]
        self.date_memory = [self.data.date.unique()[0]]

        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]

        self.covs = self.data["cov_list"].values[0]
        self.state = np.append(
            np.array(self.covs),
            [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
            axis=0)
        return self.state


    def render(self, mode="human"):
        return self.state


    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output


    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value


    def save_action_memory(self):
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
