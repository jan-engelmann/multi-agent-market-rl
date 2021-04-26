if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.abspath("."))
    print(sys.path)

import torch
import torch.multiprocessing as mp
import time
import itertools
import pandas as pd
from tqdm import tqdm

from marl_env.markets import MarketMatchHiLo
from marl_env.agents import DummyAgent
from marl_env.info_setting import BlackBoxSetting, OfferInformationSetting
from gym.spaces import Box
import numpy as np


def get_agent_actions(agent, observation):
    return agent.get_action(observation)


class MultiAgentEnvironment:
    def __init__(
        self,
        sellers,
        buyers,
        market,
        info_setting,
        n_environments,
        worker_pool,
    ):

        self.n_sellers = len(sellers)
        self.n_buyers = len(buyers)
        self.n_agents = self.n_sellers + self.n_buyers
        self.max_n_deals = min(self.n_buyers, self.n_sellers)
        self.max_group_size = max(self.n_buyers, self.n_sellers)
        self.n_environments = n_environments  # batch_size

        self.sellers = sellers
        self.buyers = buyers

        self.market = market
        self.info_setting: OfferInformationSetting = info_setting

        self.worker_pool = worker_pool

        self.observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=self.info_setting.observation_space.shape
        )
        self.reset()

    def reset(self):
        self.s_reservations = torch.rand(self.n_sellers)
        self.b_reservations = torch.rand(self.n_buyers)
        self.past_actions = []
        self.observations = []

    def get_actions(self):
        # TODO: improve distributed calculation
        agent_observation_tuples = [
            (agent, obs)
            for agent, obs in zip(
                itertools.chain(self.sellers, self.buyers), self.observations[-1]
            )
        ]
        res = torch.stack(
            self.worker_pool.starmap(
                get_agent_actions,
                agent_observation_tuples,
            )
        ).T
        return torch.split(res, [self.n_sellers, self.n_buyers], dim=1)

    def calculate_rewards(self, deals_sellers, deals_buyers):
        rewards_sellers = deals_sellers - self.s_reservations[None, :]
        rewards_buyers = self.b_reservations[None, :] - deals_buyers
        return rewards_sellers, rewards_buyers

    def store_observations(self):
        # TODO: actually implement valid observations
        self.observations.append(self.info_setting.get_states(self.market))

    def step(self):
        self.store_observations()
        s_actions, b_actions = self.get_actions()
        deals_sellers, deals_buyers = self.market.step(s_actions, b_actions)
        rewards_sellers, rewards_buyers = self.calculate_rewards(
            deals_sellers, deals_buyers
        )

        # print(deals_buyers, deals_sellers, rewards_buyers, rewards_sellers, sep="\n")


if __name__ == "__main__":
    n_sellers = 50
    n_buyers = 50
    n_environments = 15
    n_processes = [1]  # [1, 2, 4, 6, 8]
    times = []
    n_iterations = 1
    n_steps = 50

    sellers = [
        DummyAgent(idx, num)
        for idx, num in enumerate(torch.rand(n_sellers, n_environments))
    ]
    buyers = [
        DummyAgent(idx, num)
        for idx, num in enumerate(torch.rand(n_buyers, n_environments))
    ]
    buyer_ids = [agent.id for agent in buyers]
    seller_ids = [agent.id for agent in sellers]
    market = MarketMatchHiLo(buyer_ids, seller_ids, n_environments, max_steps=30)
    # info_setting = OfferInformationSetting(n_offers=3)
    info_setting = BlackBoxSetting()
    for n_proc in tqdm(n_processes, desc="n_processes"):
        for it in tqdm(range(n_iterations), desc="n_iterations", leave=False):
            duration = 0.0
            start = time.time()
            with mp.Pool(n_proc) as pool:
                env = MultiAgentEnvironment(
                    sellers,
                    buyers,
                    market,
                    info_setting,
                    n_environments,
                    pool,
                )
                for i in range(n_steps):
                    env.step()
            end = time.time()
            duration += end - start
        times.append(duration / n_iterations)  # type: ignore

    df = pd.DataFrame(dict(n_processes=n_processes, time=times))
    print(df)
    print("Finished")
