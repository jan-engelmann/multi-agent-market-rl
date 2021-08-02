import torch


class RewardSetting:
    """
    Abstract reward setting class
    """

    def __init__(self, env):
        self.env = env

    def seller_reward(self, seller_deals):
        pass

    def buyer_reward(self, buyer_deals):
        pass


class NoDealPenaltyReward(RewardSetting):
    """
    Punishes buyers with a linearly increasing negative reward if they do not manage to close a deal after N rounds.

    Parameters
    ----------
    kwargs:
        no_deal_max: int, optional (default=10)
            Number of allowed time steps without making a deal before being punished
    """

    def __init__(self, env, **kwargs):
        self.no_deal_max = kwargs.pop("no_deal_max", 10)
        super(NoDealPenaltyReward, self).__init__(env)

    def seller_reward(self, seller_deals):
        """

        Parameters
        ----------
        seller_deals: torch.tensor
            Has shape (n_sellers,)

        Returns
        -------
        rew: torch.tensor
            Has shape (n_sellers)
        """
        rew = seller_deals - self.env.s_reservations
        rew = torch.mul(rew, ~self.env.done_sellers)
        return rew

    def buyer_reward(self, buyer_deals):
        """

        Parameters
        ----------
        buyer_deals: torch.tensor
            Has shape (n_buyers)

        Returns
        -------
        rew: torch.tensor
            Has shape (n_buyers)
        """
        done_buyers = self.env.done_buyers + self.env.newly_finished_buyers
        no_deal_penalty = -(
            1 + torch.max(torch.tensor([self.env.market.time - self.no_deal_max, -1], device=self.env.env_device))
        )
        no_deal_penalty = torch.mul(no_deal_penalty, ~done_buyers)

        rew = torch.mul(
            self.env.b_reservations - buyer_deals, self.env.newly_finished_buyers
        )
        rew = torch.add(rew, no_deal_penalty)
        return rew
