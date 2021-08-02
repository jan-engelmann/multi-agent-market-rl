import torch
import typing


class BaseMarketEngine:
    def __init__(self, n_sellers, n_buyers, device=torch.device('cpu'), **kwargs):
        """

        Parameters
        ----------
        n_sellers: int
            Number of agent sellers
        n_buyers: int
            Number of agent buyers
        device: torch.device
            Allows to allocate the market engine to a cpu or gpu device
        kwargs: optional
            max_steps (default=30), max time steps of one episode/game
        """
        self.n_sellers = n_sellers
        self.n_buyers = n_buyers
        self.n_agents = n_sellers + n_buyers
        self.max_group_size = max(n_buyers, n_sellers)
        self.max_n_deals = min(n_buyers, n_sellers)

        self.max_steps = kwargs.pop("max_steps", 30)

        self.time = 0
        self.buyer_history = list()
        self.seller_history = list()
        self.deal_history = list()

        self.device = device

    def reset(self):
        """Reset the market to its initial unmatched state."""
        self.time = 0
        self.buyer_history.clear()
        self.seller_history.clear()
        self.deal_history.clear()

    def calculate_deals(
        self, s_actions: torch.Tensor, b_actions: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        raise NotImplementedError

    def step(
        self, s_actions: torch.Tensor, b_actions: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Does a market step appending actions and deals to history and forwarding market time

        Parameters
        ----------
        s_actions : torch.Tensor
            Tensor of seller actions, shape (n_sellers,)
        b_actions : torch.Tensor
            Tensor of buyer actions, shape (n_buyers,)

        Returns
        -------
        typing.Tuple[torch.Tensor torch.Tensor]
            (deals_sellers, deals_buyers) with shapes (n_sellers,), (n_buyers,)
        """
        self.seller_history.append(s_actions)
        self.buyer_history.append(b_actions)
        deals_sellers, deals_buyers = self.calculate_deals(s_actions, b_actions)

        self.time += 1

        return deals_sellers, deals_buyers


class MarketMatchHiLo(BaseMarketEngine):
    """
    Market engine using mechanism that highest buying offer is matched with lowest selling offer
    """

    def __init__(self, n_sellers, n_buyers, device=torch.device('cpu'), **kwargs):
        super(MarketMatchHiLo, self).__init__(n_sellers, n_buyers, device=device, **kwargs)

    def calculate_deals(self, s_actions, b_actions):
        # sort actions of sellers and buyers
        # using mechanism that highest buying offer is matched with lowest selling offer
        s_actions_sorted, s_actions_indices = s_actions.sort()
        b_actions_sorted, b_actions_indices = b_actions.sort(descending=True)

        # get mask for all deals that happen
        bid_offer_diffs = torch.zeros(self.max_group_size, device=self.device)
        bid_offer_diffs[: self.max_n_deals] = (
            b_actions_sorted[: self.max_n_deals] - s_actions_sorted[: self.max_n_deals]
        )
        no_deal_mask = (
            bid_offer_diffs <= 0
        )  # if true => no deal, if the bid is lower than the offer no deal happens
        s_realized_sorted_deals = s_actions_sorted.clone()
        s_realized_sorted_deals = torch.mul(
            s_realized_sorted_deals, ~no_deal_mask[: self.n_sellers]
        )

        b_realized_sorted_deals = b_actions_sorted.clone()
        b_realized_sorted_deals = torch.mul(
            b_realized_sorted_deals, ~no_deal_mask[: self.n_buyers]
        )

        # calculating deal prices
        sorted_deal_prices = torch.zeros(self.max_group_size, device=self.device)
        sorted_deal_prices[: self.max_n_deals] = (
            b_realized_sorted_deals[: self.max_n_deals]
            + s_realized_sorted_deals[: self.max_n_deals]
        ) / 2.0
        self.deal_history.append(sorted_deal_prices)

        # getting the realized prices in the original ordering for buyers&sellers, no deal means 0
        deals_buyers_original = sorted_deal_prices[: self.n_buyers].gather(
            0, b_actions_indices.argsort().squeeze()
        )

        deals_sellers_original = sorted_deal_prices[: self.n_sellers].gather(
            0, s_actions_indices.argsort().squeeze()
        )

        return deals_sellers_original, deals_buyers_original
