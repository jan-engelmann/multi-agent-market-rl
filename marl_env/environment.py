if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.abspath("."))

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
    """
    TODO: !!! Rethink which tensors need to be part of autograph and which tensors can be detached !!!
    """
    def __init__(
        self,
        sellers,
        buyers,
        market,
        info_setting,
        n_environments,
    ):
        """
        TODO: Add some documentation...
        Parameters
        ----------
        sellers
        buyers
        market
        info_setting
        n_environments
        """

        self.n_sellers = len(sellers)
        self.n_buyers = len(buyers)
        self.n_agents = self.n_sellers + self.n_buyers
        self.max_n_deals = min(self.n_buyers, self.n_sellers)
        self.max_group_size = max(self.n_buyers, self.n_sellers)
        self.n_environments = n_environments  # batch_size

        self.sellers = sellers
        self.buyers = buyers
        self.all_agents = sellers + buyers

        self.market = market
        self.info_setting: OfferInformationSetting = info_setting

        # TODO: check if this observation space is correct
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=self.info_setting.observation_space.shape
        )
        self.reset()

    def reset(self):
        with torch.no_grad():
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
        # This is done by multiplying elementwise with the inverse of the masking matrix inorder to
        # not make use of inplace operations (I hope...)
        # TODO: Rethink if this is logical...
        rewards_sellers = torch.mul(rewards_sellers, ~self.done_sellers)
        rewards_buyers = torch.mul(rewards_buyers, ~self.done_buyers)

        return rewards_sellers, rewards_buyers

    def store_observations(self):
        # TODO: actually implement valid observations
        self.observations.append(self.info_setting.get_states(self.market))

    def step(self):
        # Update the mask keeping track of which agents are done in the current game.
        # This is done with the mask computed in the previous round. Since only agents who were finished since the
        # previous round should get a zero reward.
        with torch.no_grad():
            self.done_sellers += self.newly_finished_sellers
            self.done_buyers += self.newly_finished_buyers

        self.store_observations()
        s_actions, b_actions = self.get_actions()

        # Mask seller and buyer actions for agents which are already done
        # We set the asking price of sellers who are done to max(b_reservations) + 1
        # and the biding price of buyers who are done to zero
        # This results in actions not capable of producing a deal and therefore not interfering with other agents.
        # We make use of element wise multiplication with masking tensors in order to prevent inplace
        # operations (we hope...)
        with torch.no_grad():
            s_mask_val = self.b_reservations.max() + 1.0
            b_mask_val = torch.FloatTensor([0])
        s_actions = torch.mul(s_mask_val, self.done_sellers) + torch.mul(s_actions, ~self.done_sellers)
        b_actions = torch.mul(b_mask_val, self.done_buyers) + torch.mul(b_actions, ~self.done_buyers)

        deals_sellers, deals_buyers = self.market.step(s_actions, b_actions)

        with torch.no_grad():
            self.newly_finished_sellers = deals_sellers > 0
            self.newly_finished_buyers = deals_buyers > 0

        rewards_sellers, rewards_buyers = self.calculate_rewards(
            deals_sellers, deals_buyers
        )
        # TODO: break if no more deals can be made!

        return (
            self.observations[-1],
            (rewards_sellers, rewards_buyers),
            (s_actions, b_actions),
        )


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
    n_buyers = len(buyer_ids)
    n_sellers = len(seller_ids)
    market = MarketMatchHiLo(n_buyers, n_sellers, n_environments, max_steps=30)
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
