if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.abspath("."))

import torch
import itertools

from marl_env.info_setting import (
    OfferInformationSetting,
)


def get_agent_actions(agent, observation, epsilon, random_action):
    """

    Parameters
    ----------
    agent: Abstract agent class
    observation: torch.Tensor
    epsilon: float
        The probability of returning a random action in epsilon-greedy exploration
    random_action: bool
        If true action is drawn from a uniform random policy or from a specifically implemented random policy called
        'random_action'

    Returns
    -------
    action: torch.Tensor
    """
    if not random_action:
        action = agent.get_action(observation, epsilon)
    else:
        try:
            action = agent.random_action(observation, epsilon)
        except NotImplementedError:
            action = agent.get_action(observation, 1.0)
    return action


class MultiAgentEnvironment:
    def __init__(
        self,
        sellers,
        buyers,
        s_reservations,
        b_reservations,
        market,
        info_setting,
        exploration_setting,
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

        self.s_reservations = s_reservations
        self.b_reservations = b_reservations

        self.sellers = sellers
        self.buyers = buyers
        self.all_agents = sellers + buyers

        self.market = market
        self.info_setting: OfferInformationSetting = info_setting
        self.exploration_setting = exploration_setting

        self.random_action = False
        self.done = False

        self.reset()

    def reset(self):
        self.done = False
        self.market.reset()

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
            (agent, obs, self.exploration_setting.epsilon, self.random_action)
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
        rewards_sellers = torch.mul(rewards_sellers, ~self.done_sellers)
        rewards_buyers = torch.mul(rewards_buyers, ~self.done_buyers)

        return rewards_sellers, rewards_buyers

    def store_observations(self):
        # TODO: actually implement valid observations
        self.observations.append(self.info_setting.get_states(self.market))

    def step(self, random_action=False):
        """

        Parameters
        ----------
        random_action

        Returns
        -------
        current_observations: torch.Tensor
            All agent observations at the current time step t. The zero dimension has size self.n_agents
            current_observations[:n_sellers,:,:] contains all observations for the seller agents
            current_observations[n_sellers:,:,:] contains all observations for the buyer agents
        current_actions: torch.Tensor
            All agent actions at the current time step t. The last dimension has size self.n_agents
            current_actions[:, :n_sellers] contains all actions for the seller agents
            current_actions[: , n_sellers:] contains all actions for the buyer agents
        current_rewards: torch.Tensor
            All agent rewards at the current time step t. The last dimension has size self.n_agents
            current_rewards[:, :n_sellers] contains all rewards for the seller agents
            current_rewards[: , n_sellers:] contains all rewards for the buyer agents
        next_observation: torch.Tensor
            All agent observations at the next time step t + 1. The zero dimension has size self.n_agents
            next_observation[:n_sellers,:,:] contains all observations for the seller agents
            next_observation[n_sellers:,:,:] contains all observations for the buyer agents
        self.done: bool
            True if all agents are done trading or if the the game has come to an end
        """
        self.random_action = random_action

        self.store_observations()
        s_actions, b_actions = self.get_actions()

        # Mask seller and buyer actions for agents which are already done
        # We set the asking price of sellers who are done to max(b_reservations)
        # and the biding price of buyers who are done to min(s_reservations)
        # This results in actions not capable of producing a deal and therefore not interfering with other agents.
        # We make use of element wise multiplication with masking tensors in order to prevent inplace
        # operations (we hope...)
        with torch.no_grad():
            s_mask_val = self.b_reservations.max()
            b_mask_val = self.s_reservations.min()
        s_actions = torch.mul(s_mask_val, self.done_sellers) + torch.mul(s_actions, ~self.done_sellers)
        b_actions = torch.mul(b_mask_val, self.done_buyers) + torch.mul(b_actions, ~self.done_buyers)

        deals_sellers, deals_buyers = self.market.step(s_actions, b_actions)

        with torch.no_grad():
            self.newly_finished_sellers = deals_sellers > 0
            self.newly_finished_buyers = deals_buyers > 0

        rewards_sellers, rewards_buyers = self.calculate_rewards(
            deals_sellers, deals_buyers
        )

        # Update the exploration value epsilon.
        self.exploration_setting.update()

        # Update the mask keeping track of which agents are done in the current game.
        # This is done with the mask computed in the previous round. Since only agents who were finished since the
        # previous round should get a zero reward.
        self.done_sellers += self.newly_finished_sellers
        self.done_buyers += self.newly_finished_buyers

        if torch.all(self.done_sellers) or torch.all(self.done_buyers):
            self.done = True
        elif self.market.time == self.market.max_steps:
            self.done = True

        current_observations = self.observations[-1]
        current_actions = torch.cat([s_actions, b_actions], dim=-1)
        current_rewards = torch.cat([rewards_sellers, rewards_buyers], dim=-1)

        next_observation = self.info_setting.get_states(self.market)

        return (
            current_observations,
            current_actions,
            current_rewards,
            next_observation,
            self.done
        )
