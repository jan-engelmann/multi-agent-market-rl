import torch
import typing


class BaseMarketEngine:
    def __init__(self, buyer_ids, seller_ids, n_environments, max_steps=30):
        self.buyer_ids = set(buyer_ids)
        self.n_buyers = len(self.buyer_ids)
        self.seller_ids = set(seller_ids)
        self.n_sellers = len(self.seller_ids)
        self.n_agent_ids = self.n_sellers + self.n_buyers
        self.max_steps = max_steps
        self.max_group_size = max(self.n_buyers, self.n_sellers)
        self.max_n_deals = min(self.n_buyers, self.n_sellers)
        self.n_environments = n_environments
        self.reset()

    def reset(self):
        """Reset the market to its initial unmatched state."""
        self.time = 0
        self.buyer_history = list()
        self.seller_history = list()
        self.deal_history = list()

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
            Tensor of seller actions, shape n_environments, n_sellers)
        b_actions : torch.Tensor
            Tensor of buyer actions, shape (n_environments, n_buyers)

        Returns
        -------
        typing.Tuple[torch.Tensor torch.Tensor]
            (deals_sellers, deals_buyers) with shapes (n_environments, n_sellers), (n_environments, n_buyers)
        """
        self.seller_history.append(s_actions)
        self.buyer_history.append(b_actions)
        deals_sellers, deals_buyers = self.calculate_deals(s_actions, b_actions)

        self.time += 1

        return deals_sellers, deals_buyers


class MarketMatchHiLo(BaseMarketEngine):
    def __init__(self, buyer_ids, seller_ids, n_environments, max_steps=30):
        super(MarketMatchHiLo, self).__init__(
            buyer_ids, seller_ids, n_environments, max_steps=30
        )

    def calculate_deals(self, s_actions, b_actions):
        # sort actions of sellers and buyers
        # using mechanism that highest buying offer is matched with lowest selling offer
        s_actions_sorted, s_actions_indices = s_actions.sort()
        b_actions_sorted, b_actions_indices = b_actions.sort(descending=True)

        # get mask for all deals that happen
        bid_offer_diffs = torch.zeros(s_actions.shape[0], self.max_group_size)
        bid_offer_diffs[:, : self.max_n_deals] = (
            b_actions_sorted[:, : self.max_n_deals]
            - s_actions_sorted[:, : self.max_n_deals]
        )
        no_deal_mask = (
            bid_offer_diffs <= 0
        )  # if true => no deal, if the bid is lower than the offer no deal happens

        s_realized_sorted_deals = s_actions_sorted.clone()
        s_realized_sorted_deals[no_deal_mask[:, : self.n_sellers]] = 0

        b_realized_sorted_deals = b_actions_sorted.clone()
        b_realized_sorted_deals[no_deal_mask[:, : self.n_buyers]] = 0

        # calculating deal prices
        sorted_deal_prices = torch.zeros(s_actions.shape[0], self.max_group_size)
        sorted_deal_prices[:, : self.max_n_deals] = (
            b_realized_sorted_deals[:, : self.max_n_deals]
            + s_realized_sorted_deals[:, : self.max_n_deals]
        ) / 2.0
        self.deal_history.append(sorted_deal_prices)

        # getting the realized prices in the original ordering for buyers&sellers, no deal means 0
        deals_buyers_orginal = sorted_deal_prices[:, : self.n_buyers].gather(
            1, b_actions_indices.argsort(1)
        )

        deals_sellers_orginal = sorted_deal_prices[:, : self.n_sellers].gather(
            1, s_actions_indices.argsort(1)
        )

        return deals_sellers_orginal, deals_buyers_orginal