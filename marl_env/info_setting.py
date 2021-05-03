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

    def get_states(self, market):
        n_envs = market.n_environments
        n_agents = market.n_agents
        n_sellers = market.n_sellers
        total_info = torch.zeros(n_agents, n_envs, 1)
        if not (market.buyer_history or market.seller_history):
            return total_info

        b_actions = market.buyer_history[-1].T
        s_actions = market.seller_history[-1].T

        total_info[:n_sellers, :, 0] = s_actions
        total_info[n_sellers:, :, 0] = b_actions

        # Return total_info as tensor with shape (n_agents, n_envs, n_features) where n_features == 1
        # Observations are ordered in the same way as res in MultiAgentEnvironment.get_actions().
        # total_info[:n_sellers,:,:] contains all observations for the seller agents
        # total_info[n_sellers:,:,:] contains all observations for the buyer agents
        # total_info is used as input for the NN --> Should bee the leaf of the gradient graph --> Make use of .detach()
        return total_info.clone().detach()


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
          Look at observation_space, does it still have the shape (2, n_offers)?

          --> If we look at environments as being batches, it makes sense for an agent to have observations over all
              environments. You can think of it as the agent playing all environments at the same time.
              I imagine something like this:

              agent(input_tensor(n_environments, n_features)) --> action_tensor(n_environments, 1)

              So the agent takes action in all environments in one go. In the same way as a Neural Network would do when
              making use of batch stochastic gradient descent

          --> Different point:
              market.n_buyers and market.n_sellers need to be integers. Currently MarketMatchHiLo(...) gets initialised
              with lists of buyer and seller ids. Is this intentional?
    """

    def __init__(self, n_offers=5):
        self.n_offers = n_offers
        self.observation_space = Box(low=0, high=np.infty, shape=[2 * n_offers])

    def get_states(self, market):
        assert self.n_offers <= market.n_buyers
        assert self.n_offers <= market.n_sellers

        n = self.n_offers
        n_envs = market.n_environments
        n_agents = market.n_agents
        total_info = torch.zeros(n_agents, n_envs, 2, n)
        if not (market.buyer_history or market.seller_history):
            # Return total_info as tensor with shape (n_agents, n_envs, n_features) where n_features == 2 * n_offers
            return total_info.contiguous().view(n_agents, n_envs, -1)

        # Each history contains a list of tensors of shape (n_environments, n_agents) for sellers and buyers
        # respectively
        b_actions = market.buyer_history[-1]
        s_actions = market.seller_history[-1]

        # sort the buyer and seller actions inorder to find the N best offers of either side.
        # Best: seller --> lowest | buyer --> highest
        s_actions_sorted = s_actions.sort()[0][:, :n]
        b_actions_sorted = b_actions.sort(descending=True)[0][:, :n]

        total_info[:, :, 0, :] = b_actions_sorted.unsqueeze_(0).expand(
            n_agents, n_envs, n
        )
        total_info[:, :, 1, :] = s_actions_sorted.unsqueeze_(0).expand(
            n_agents, n_envs, n
        )

        # The information each agent gets is the same
        # Return total_info as tensor with shape (n_agents, n_envs, n_features) where n_features == 2 * n_offers
        total_info = total_info.contiguous().view(n_agents, n_envs, -1).clone().detach()

        return total_info


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

    def get_states(self, market):
        n = self.n_deals
        n_envs = market.n_environments
        n_agents = market.n_agents
        total_info = torch.zeros(n_agents, n_envs, n)

        if not market.deal_history:
            return total_info

        # Get the best n best deals from the last round.
        # deal_history is already sorted.
        total_info = (
            market.deal_history[-1][:, :n].unsqueeze_(0).expand(n_agents, n_envs, n)
        ).clone().detach()
        return total_info


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
        self.observation_space = Box(
            low=0, high=np.infty, shape=[base_setting.observation_space.shape[0] + 1]
        )

    def get_states(self, market):
        n_envs = market.n_environments
        n_agents = market.n_agents
        base_obs = self.base_setting.get_states(market)

        # We want zero to indicate the initial state of the market and 1 to indicate the end state of the market.
        normalized_time = market.time / self.max_steps
        # TODO: put this together with environment time constraints
        assert normalized_time <= 1  # otherwise time constraint violated

        time_info = torch.full((n_agents, n_envs, 1), normalized_time)
        total_info = torch.cat((base_obs, time_info), -1).clone().detach()

        # Return total_info with one added feature representing the current normalized market time.
        return total_info
