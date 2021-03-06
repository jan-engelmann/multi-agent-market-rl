# Assure that a / b is mapped to __truediv__() for all python versions >= 2.1
# If a / b is intended as __floordiv__(), make use of // operator
from __future__ import division

import torch
import inspect


class InformationSetting:
    """
    Abstract information setting class.

    Attributes
    ----------
    env: Environment object
            The current environment class object.
    """

    def __init__(self, env):
        """

        Parameters
        ----------
        env: environment class object
        """
        self.env = env

    def get_states(self):
        """
        Compute the observations of agents given the market object.

        """
        pass


class BlackBoxSetting(InformationSetting):
    """
    The agent is aware of only its own last offer.

    """

    def __init__(self, env, **kwargs):
        """

        Parameters
        ----------
        env: environment class object
        kwargs:
            Additional keyword arguments
        """
        super(BlackBoxSetting, self).__init__(env)

    def get_states(self):
        """

        Returns
        -------
        total_info: torch.tensor
            Return total_info as tensor with shape (n_agents, n_features) where n_features == 1
            Observations are ordered in the same way as res in MultiAgentEnvironment.get_actions().
            total_info[:n_sellers, :] contains all observations for the seller agents
            total_info[n_sellers:, :] contains all observations for the buyer agents
        """
        n_agents = self.env.market.n_agents
        n_sellers = self.env.market.n_sellers
        done_sellers = self.env.done_sellers.clone()
        done_buyers = self.env.done_buyers.clone()
        states = (
            torch.cat([done_sellers, done_buyers], dim=-1).unsqueeze(0).transpose(0, 1)
        )
        total_info = torch.zeros((n_agents, 1), device=self.env.env_device)
        total_info = torch.cat([total_info, states], dim=-1)
        if not (self.env.market.buyer_history or self.env.market.seller_history):
            return total_info

        b_actions = self.env.market.buyer_history[-1]
        s_actions = self.env.market.seller_history[-1]

        total_info[:n_sellers, 0] = s_actions
        total_info[n_sellers:, 0] = b_actions

        # Return total_info as tensor with shape (n_agents, n_features) where n_features == 1
        # Observations are ordered in the same way as res in MultiAgentEnvironment.get_actions().
        # total_info[:n_sellers, :] contains all observations for the seller agents
        # total_info[n_sellers:, :] contains all observations for the buyer agents
        # total_info is used as input for the NN --> Should bee the leaf of the gradient graph --> Make use of .detach()
        return total_info.clone().detach()


class OfferInformationSetting(InformationSetting):
    """
    The agent is aware of the best N offers of either side of the last round.

    Parameters
    ----------
    kwargs: dict
        n_offers: int, optional (default=1)
            Number of offers to see. For instance, 5 would mean the agents see the
            best 5 bids and asks.

    """

    def __init__(self, env, **kwargs):
        """

        Parameters
        ----------
        env: environment class object
        kwargs:
            'n_offers' (int)
        """
        self.n_offers = kwargs.pop("n_offers", 1)
        super(OfferInformationSetting, self).__init__(env)

    def get_states(self):
        """

        Returns
        -------
        total_info: torch.tensor
            Return total_info as tensor with shape (n_agents, n_features) where n_features == 2*n_offers
            Observations are ordered in the same way as res in MultiAgentEnvironment.get_actions().
            total_info[:n_sellers, :] contains all observations for the seller agents
            total_info[n_sellers:, :] contains all observations for the buyer agents
        """
        assert self.n_offers <= self.env.market.n_buyers
        assert self.n_offers <= self.env.market.n_sellers

        n = self.n_offers
        n_agents = self.env.market.n_agents
        done_sellers = self.env.done_sellers.clone()
        done_buyers = self.env.done_buyers.clone()
        states = (
            torch.cat([done_sellers, done_buyers], dim=-1).unsqueeze(0).transpose(0, 1)
        )
        total_info = torch.zeros((n_agents, 2, n), device=self.env.env_device)
        if not (self.env.market.buyer_history or self.env.market.seller_history):
            # Return total_info as tensor with shape (n_agents, n_features) where n_features == 2 * n_offers
            total_info = total_info.contiguous().view(n_agents, -1)
            total_info = torch.cat([total_info, states], dim=-1)
            return total_info

        # Each history contains a list of tensors of shape (n_agents,) for sellers and buyers
        # respectively
        b_actions = self.env.market.buyer_history[-1]
        s_actions = self.env.market.seller_history[-1]

        # sort the buyer and seller actions inorder to find the N best offers of either side.
        # Best: seller --> lowest | buyer --> highest
        s_actions_sorted = s_actions.sort()[0][:n]
        b_actions_sorted = b_actions.sort(descending=True)[0][:n]

        total_info[:, 0, :-1] = b_actions_sorted.unsqueeze(0).expand(n_agents, n)
        total_info[:, 1, :-1] = s_actions_sorted.unsqueeze(0).expand(n_agents, n)

        # The information each agent gets is the same
        # Return total_info as tensor with shape (n_agents, n_features) where n_features == 2 * n_offers
        total_info = total_info.contiguous().view(n_agents, -1).clone().detach()
        total_info = torch.cat([total_info, states], dim=-1)

        return total_info


class DealInformationSetting(InformationSetting):
    """
    The agent is aware of N deals of the last round.

    Note: the N deals need not be sorted on deal price. It depends the order
    the matcher matches deals, see ``MarketEngine.matcher``.

    Parameters
    ----------
    kwargs: dict
        n_deals: int, optional (default=1)
            Number of deals to see.

    Attributes
    ----------

    """

    def __init__(self, env, **kwargs):
        """

        Parameters
        ----------
        env: environment class object
        kwargs:
            'n_deals' (int)
        """
        self.n_deals = kwargs.pop("n_deals", 1)
        super(DealInformationSetting, self).__init__(env)

    def get_states(self):
        """

        Returns
        -------
        total_info: torch.tensor
            Return total_info as tensor with shape (n_agents, n_features) where n_features == n_deals
            Observations are ordered in the same way as res in MultiAgentEnvironment.get_actions().
            total_info[:n_sellers, :] contains all observations for the seller agents
            total_info[n_sellers:, :] contains all observations for the buyer agents
        """
        n_agents = self.env.market.n_agents
        done_sellers = self.env.done_sellers.clone()
        done_buyers = self.env.done_buyers.clone()
        states = (
            torch.cat([done_sellers, done_buyers], dim=-1).unsqueeze(0).transpose(0, 1)
        )
        total_info = torch.zeros((n_agents, self.n_deals), device=self.env.env_device)
        total_info = torch.cat([total_info, states], dim=-1)

        if not self.env.market.deal_history:
            return total_info

        # Get the N best deals from the last round.
        # deal_history is already sorted.
        total_info = (
            (
                self.env.market.deal_history[-1][: self.n_deals]
                .unsqueeze(0)
                .expand(n_agents, self.n_deals)
            )
            .clone()
            .detach()
        )
        total_info = torch.cat([total_info, states], dim=-1)
        return total_info


class TimeInformationWrapper(InformationSetting):
    """
    Wrapper to include the time in the observation.

    This class takes as input another information setting and adds the time
    of the market to the observations of that information setting. This allows
    certain fixed agents to adopt time-dependent strategies.

    Parameters
    ----------
    kwargs: dict
        base_setting: InformationSetting object
            The base information setting to add time to.

    """

    def __init__(self, env, **kwargs):
        """

        Parameters
        ----------
        env: environment class object
        kwargs:
            'base_setting' (str)
        """
        base_setting = kwargs.pop("base_setting", "BlackBoxSetting")

        if isinstance(base_setting, str):
            self.base_setting = globals()[base_setting](env, **kwargs)
        else:
            assert inspect.isclass(type(base_setting))
            self.base_setting = base_setting

    def get_states(self):
        """

        Returns
        -------
        total_info: torch.tensor
            Return total_info as tensor with shape (n_agents, n_features) where n_features == base_features + 1
            Observations are ordered in the same way as res in MultiAgentEnvironment.get_actions().
            total_info[:n_sellers, :] contains all observations for the seller agents
            total_info[n_sellers:, :] contains all observations for the buyer agents
        """
        n_agents = self.base_setting.env.market.n_agents
        base_obs = self.base_setting.get_states()

        # We want zero to indicate the initial state of the market and 1 to indicate the end state of the market.
        normalized_time = self.base_setting.env.market.time / self.base_setting.env.market.max_steps
        # TODO: put this together with environment time constraints
        assert normalized_time <= 1  # otherwise time constraint violated

        time_info = torch.full((n_agents, 1), normalized_time, device=self.base_setting.env.env_device)
        total_info = torch.cat((base_obs, time_info), -1).clone().detach()

        # Return total_info with one added feature representing the current normalized market time.
        return total_info
