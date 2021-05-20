if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.abspath("."))

import torch
import inspect
import itertools

import marl_env.markets as markets
import marl_env.info_setting as inf_setting
import marl_env.reward_setting as rew_setting
import marl_env.exploration_setting as expo_setting


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
        info_setting,
        market,
        exploration_setting,
        reward_setting,
        **kwargs
    ):
        """
        TODO: Add some documentation...
        Parameters
        ----------
        sellers
        buyers
        market
        info_setting
        """

        self.n_sellers = len(sellers)
        self.n_buyers = len(buyers)
        self.n_agents = self.n_sellers + self.n_buyers
        self.max_n_deals = min(self.n_buyers, self.n_sellers)
        self.max_group_size = max(self.n_buyers, self.n_sellers)

        self.s_reservations = s_reservations
        self.b_reservations = b_reservations

        self.sellers = sellers
        self.buyers = buyers
        self.all_agents = sellers + buyers

        if isinstance(market, str):
            self.market = getattr(markets, market)(
                len(sellers), len(buyers), **kwargs.pop("market_settings", {})
            )
        else:
            assert inspect.isclass(
                type(market)
            ), "Provide market class object or string name of market class object"
            self.market = market

        if isinstance(reward_setting, str):
            self.reward_setting = getattr(rew_setting, reward_setting)(
                self, **kwargs.pop("reward_setting", {})
            )
        else:
            assert inspect.isclass(type(reward_setting)), (
                "Provide reward class object or string " "name of reward class object"
            )
            self.reward_setting = reward_setting

        if isinstance(exploration_setting, str):
            self.exploration_setting = getattr(expo_setting, exploration_setting)(
                **kwargs.pop("exploration_setting", {})
            )
        else:
            assert inspect.isclass(type(exploration_setting)), (
                "Provide exploration class object or string name "
                "of exploration class object"
            )
            self.exploration_setting = exploration_setting

        if isinstance(info_setting, str):
            self.info_setting = getattr(inf_setting, info_setting)(
                self.market, **kwargs.pop("info_setting", {})
            )
        else:
            assert inspect.isclass(type(info_setting)), (
                "Provide information class object or string name of "
                "information class object"
            )
            self.info_setting = info_setting

        self.random_action = False
        self.done = False

        self.past_actions = list()
        self.observations = list()

        self.reset()

    def reset(self):
        self.done = False
        self.market.reset()

        # Create a mask keeping track of which agent is already done in the current game.
        self.done_sellers = torch.full((self.n_sellers,), False, dtype=torch.bool)
        self.done_buyers = torch.full((self.n_buyers,), False, dtype=torch.bool)
        self.newly_finished_sellers = self.done_sellers.clone()
        self.newly_finished_buyers = self.done_buyers.clone()

        self.past_actions.clear()
        self.observations.clear()

    def get_actions(self):
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
        ).T.squeeze()
        return torch.split(res, [self.n_sellers, self.n_buyers], dim=-1)

    def calculate_rewards(self, deals_sellers, deals_buyers):
        rewards_sellers = self.reward_setting.seller_reward(deals_sellers)
        rewards_buyers = self.reward_setting.buyer_reward(deals_buyers)

        return rewards_sellers, rewards_buyers

    def store_observations(self):
        self.observations.append(self.info_setting.get_states())

    def step(self, random_action=False):
        """

        Parameters
        ----------
        random_action

        Returns
        -------
        current_observations: torch.Tensor
            All agent observations at the current time step t. The zero dimension has size self.n_agents
            current_observations[:n_sellers, :] contains all observations for the seller agents
            current_observations[n_sellers:, :] contains all observations for the buyer agents
        current_actions: torch.Tensor
            All agent actions at the current time step t. The last dimension has size self.n_agents
            current_actions[:n_sellers] contains all actions for the seller agents
            current_actions[n_sellers:] contains all actions for the buyer agents
        current_rewards: torch.Tensor
            All agent rewards at the current time step t. The last dimension has size self.n_agents
            current_rewards[:n_sellers] contains all rewards for the seller agents
            current_rewards[n_sellers:] contains all rewards for the buyer agents
        next_observation: torch.Tensor
            All agent observations at the next time step t + 1. The zero dimension has size self.n_agents
            next_observation[:n_sellers, :] contains all observations for the seller agents
            next_observation[n_sellers:, :] contains all observations for the buyer agents
        self.done: bool
            True if all agents are done trading or if the the game has come to an end
        """
        self.random_action = random_action

        self.store_observations()
        s_actions, b_actions = self.get_actions()
        print("obs ", self.observations[-1])
        print("s_act ", s_actions)
        print("b_act ", b_actions)

        # Mask seller and buyer actions for agents which are already done
        # We set the asking price of sellers who are done to max(b_reservations)
        # and the biding price of buyers who are done to min(s_reservations)
        # This results in actions not capable of producing a deal and therefore not interfering with other agents.
        with torch.no_grad():
            s_mask_val = self.b_reservations.max()
            b_mask_val = self.s_reservations.min()
        s_actions = torch.mul(s_mask_val, self.done_sellers) + torch.mul(
            s_actions, ~self.done_sellers
        )
        b_actions = torch.mul(b_mask_val, self.done_buyers) + torch.mul(
            b_actions, ~self.done_buyers
        )

        deals_sellers, deals_buyers = self.market.step(s_actions, b_actions)
        print("s_deal ", deals_sellers)
        print("b_deal", deals_buyers)

        with torch.no_grad():
            self.newly_finished_sellers = deals_sellers > 0
            self.newly_finished_buyers = deals_buyers > 0

        rewards_sellers, rewards_buyers = self.calculate_rewards(
            deals_sellers, deals_buyers
        )
        print("s_rew ", rewards_sellers)
        print("b_rew ", rewards_buyers)

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
        print("Done ", self.done)

        current_observations = self.observations[-1]
        current_actions = torch.cat([s_actions, b_actions], dim=-1)
        current_rewards = torch.cat([rewards_sellers, rewards_buyers], dim=-1)

        next_observation = self.info_setting.get_states()

        return (
            current_observations,
            current_actions,
            current_rewards,
            next_observation,
            self.done,
        )
