# Assure that a / b is mapped to __truediv__() for all python versions >= 2.1
# If a / b is intended as __floordiv__(), make use of // operator
from __future__ import division

import torch
import numpy as np
from gym.spaces import Discrete, Box, Tuple

class InformationSetting:
    """
    Abstract information setting class.

    Attributes
    ----------
    observation_space: gym.spaces object
        The specification of the observation space under this setting.
    """

    def __init__(self):
        pass

    def get_states(self, market):
        """
        Compute the observations of agents given the market object.

        Parameters
        ----------
        market: MarketEngine object
            The current market object.

        Returns
        -------
        states: dict
            A dictionary of observations for each agent id. Each observation
            should be an element of the ``observation_space``.
        """
        pass

    def get_state(self, market):
        return self.get_states(market)


class BlackBoxSetting(InformationSetting):
    """
    The agent is aware of only its own last offer.

    Attributes
    ----------
    observation_space: Box object
        Represents the last offer of the agent. Each element is a numpy array
        with a single entry. If there was no offer, it will be ``[0]``.
    """
    def __init__(self):
        self.observation_space = Box(low=0, high=np.infty, shape=[1])

    def get_states(self, agent_ids, market):
        if not market.offer_history:
            return {agent_id: np.array([0]) for agent_id in agent_ids}

        bids, asks = market.offer_history[-1]
        # This might be a bit slow
        bids = {agent_id: val for val, agent_id in bids}
        asks = {agent_id: val for val, agent_id in asks}
        result = {}
        for agent_id in agent_ids:
            if agent_id in bids:
                result[agent_id] = np.array([bids[agent_id]])
            elif agent_id in asks:
                result[agent_id] = np.array([asks[agent_id]])
            else:
                result[agent_id] = np.array([0])
        return result


class OfferInformationSetting(InformationSetting):
    """
    The agent is aware of the best N offers of either side of the last round.

    Parameters
    ----------
    n_offers: int, optional (default=5)
        Number of offers to see. For instance, 5 would mean the agents see the
        best 5 bids and asks.

    Attributes
    ----------
    observation_space: Box object
        Each element is a numpy array of shape ``(2, n_offers)``. The first row
        contains the bids and second row the asks. No offers will be
        represented by 0.

    TODO: Think about environment number for observation_space. I don't think it makes sense to have an observation
          space over all environments. Each agent should only have the observations coming from his environment.
          But double check implementation to be sure...
          !!! market.n_environments not part of market !!!
    """
    def __init__(self, n_offers=5):
        self.n_offers = n_offers
        self.observation_space = Box(low=0, high=np.infty, shape=[2, n_offers])

    def get_states(self, market):
        n = self.n_offers
        n_envs = market.n_environments
        n_agents = len(market.agent_ids)
        total_info = torch.zeros(n_agents, n_envs, 2, n)
        if not (market.buyer_history or market.seller_history):
            # Return total_info as tensor with shape (n_agents, n_envs, n_features) where n_features == 2 * n_offers
            return total_info.contiguous().view(n_agents, n_envs, -1)

        # Each history contains a list of tensors of shape (n_agents, n_environments) for sellers and buyers
        # respectively
        b_actions = market.buyer_history[-1].T
        s_actions = market.seller_history[-1].T

        # sort the buyer and seller actions inorder to find the N best offers of either side.
        # Best: seller --> lowest | buyer --> highest
        s_actions_sorted, _ = s_actions.sort()[:, 0:n]
        b_actions_sorted, _ = b_actions.sort(descending=True)[:, 0:n]

        total_info[:, :, 0, :] = b_actions_sorted.unsqueeze_(0).expand(n_agents, n_envs, n)
        total_info[:, :, 1, :] = s_actions_sorted.unsqueeze_(0).expand(n_agents, n_envs, n)

        # The information each agent gets is the same
        # Return total_info as tensor with shape (n_agents, n_envs, n_features) where n_features == 2 * n_offers
        return total_info.contiguous().view(n_agents, n_envs, -1)


class DealInformationSetting(InformationSetting):
    """
    The agent is aware of N deals of the last round.

    Note: the N deals need not be sorted on deal price. It depends the order
    the matcher matches deals, see ``MarketEngine.matcher``.

    Parameters
    ----------
    n_deals: int, optional (default=5)
        Number of deals to see.

    Attributes
    ----------
    observation_space: Box object
        Each element is a numpy array of shape ``(n_offers,)``. No deals will
        be represented by 0.
    """
    def __init__(self, n_deals=5):
        self.n_deals = n_deals
        self.observation_space = Box(low=0, high=np.infty, shape=[n_deals])

    def get_states(self, agent_ids, market):
        n = self.n_deals
        if market.deal_history:
            # Here we exploit that deal_history contains the same deal twice
            # in a row, once for the buyer and once for the seller. Since
            # Python >= 3.6 dicts preserve the order of insertion, we can
            # rely on this to obtain the distinct deals that happened.
            deals = list(market.deal_history[-1].values())[0:2*n:2]
            deals = np.pad(deals, (0, n-len(deals))) # Pad it with zeros
        else: deals = np.zeros(n)
        return {agent_id: deals for agent_id in agent_ids}


class TimeInformationWrapper(InformationSetting):
    """
    Wrapper to include the time in the observation.

    This class takes as input another information setting and adds the time
    of the market to the observations of that information setting. This allows
    certain fixed agents to adopt time-dependent strategies.

    Parameters
    ----------
    base_setting: InformationSetting object
        The base information setting to add time to.
    max_steps: int, optional (default=30)
        This should be the same as the ``max_steps`` parameter in the market
        engine, as it determines the maximum number of time steps there can be.

    """
    def __init__(self, base_setting, max_steps=30):
        self.base_setting = base_setting
        self.max_steps = max_steps
        self.observation_space = Tuple((base_setting.observation_space,
                                        Discrete(max_steps)))

    def get_states(self, agent_ids, market):
        base_obs = self.base_setting.get_states(agent_ids, market)

        # We want zero to indicate the initial state of the market and 1 to indicate the end state of the market.
        normalized_time = market.time / self.max_steps
        result = {}
        for agent_id, obs in base_obs.items():
            result[agent_id] = (obs, normalized_time)
        return result