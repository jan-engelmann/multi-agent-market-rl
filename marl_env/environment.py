if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.abspath("."))
    print(sys.path)

import gym
import torch
import torch.multiprocessing as mp
import time
import itertools
import pandas as pd
from tqdm import tqdm

from marl_env.markets import MarketMatchHiLo
from marl_env.agents import DummyAgent
from marl_env.info_setting import (
    BlackBoxSetting,
    OfferInformationSetting,
    TimeInformationWrapper,
)
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

        # TODO: check if this observation space is correct
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=self.info_setting.observation_space.shape
        )
        self.reset()

    def reset(self):
        self.s_reservations = torch.rand(self.n_sellers)
        self.b_reservations = torch.rand(self.n_buyers)

        # Create a mask keeping track of which agent is already done in the current game.
        self.done_sellers = torch.full(
            (self.n_environments, self.n_sellers), False, dtype=torch.bool
        )
        self.done_buyers = torch.full(
            (self.n_environments, self.n_buyers), False, dtype=torch.bool
        )
        self.newly_finished_sellers = self.done_sellers.clone()
        self.newly_finished_buyers = self.done_buyers.clone()

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
            list(
                itertools.starmap(
                    get_agent_actions,
                    agent_observation_tuples,
                )
            )
        ).T
        return torch.split(res, [self.n_sellers, self.n_buyers], dim=1)

    def calculate_rewards(self, deals_sellers, deals_buyers):
        rewards_sellers = deals_sellers - self.s_reservations[None, :]
        rewards_buyers = self.b_reservations[None, :] - deals_buyers

        # Agents who are finished since the previous round receive a zero reward.
        # TODO: Rethink if this is logical...
        rewards_sellers[self.done_sellers] = 0
        rewards_buyers[self.done_buyers] = 0

        return rewards_sellers, rewards_buyers

    def store_observations(self):
        # TODO: actually implement valid observations
        self.observations.append(self.info_setting.get_states(self.market))

    def step(self):
        # Update the mask keeping track of which agents are done in the current game.
        # This is done with the mask computed in the previous round. Since only agents who were finished since the
        # previous round should get a zero reward.
        self.done_sellers += self.newly_finished_sellers
        self.done_buyers += self.newly_finished_buyers

        self.store_observations()
        s_actions, b_actions = self.get_actions()

        # Mask seller and buyer actions for agents which are already done
        # We set the asking price of sellers who are done to max(b_reservations) + 1
        # and the biding price of buyers who are done to zero
        # This results in actions not capable of producing a deal and therefore not interfering with other agents.
        s_actions[self.done_sellers] = self.b_reservations.max() + 1
        b_actions[self.done_buyers] = 0

        deals_sellers, deals_buyers = self.market.step(s_actions, b_actions)
        self.newly_finished_sellers = deals_sellers > 0
        self.newly_finished_buyers = deals_buyers > 0

        rewards_sellers, rewards_buyers = self.calculate_rewards(
            deals_sellers, deals_buyers
        )
        print("made one step")

        return rewards_sellers, rewards_buyers


if __name__ == "__main__":
    n_sellers = 5
    n_buyers = 5
    n_environments = 2
    n_processes = [1]  # [1, 2, 4, 6, 8]
    times = []
    n_iterations = 1
    n_steps = 5

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
    info_setting = TimeInformationWrapper(OfferInformationSetting())

    env = MultiAgentEnvironment(
        sellers,
        buyers,
        market,
        info_setting,
        n_environments,
    )
    for i in range(n_steps):
        env.step()

    print("stop")
