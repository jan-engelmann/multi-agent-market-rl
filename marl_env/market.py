import torch


class MarketEngine:
    """
    Core double auction, single unit market matching enigne

    Parameters
    ----------
    buyers : array_like
        List of buyer ids, should be distinct.

    sellers : array_like
        List of seller ids, should be distinct.

    max_steps: int (optional, default=30)
        Number of maximum market rounds.


    Attributes
    -------
    time: int
        The current time step the market is in. Starts at 0.

    done: set
        The set of agent ids that are done for this game. If ``max_steps`` is
        reached, this will contain all agent ids.

    offer_history: list
        This is a list of all offers up until the current step. Each entry of
        this list is a tuple of the form ``(bids, asks)`` which represent the
        set of bids and asks respectively for that time step. Both ``bids`` and
        ``asks`` are in the form used by the matcher, i.e., they are lists of
        tuples of the form ``[(offer1, agent_id1), (offer2, agent_id2), ...]``.

    deal_history: list
        A list of all deals up until the current time step. Each entry contains
        a dict of the form ``{agent_id1: deal_price1, ...}`` for all agents
        that were matched in that round.
    """

    def __init__(self, buyer_ids, seller_ids, max_steps=30):
        self.buyer_ids = set(buyer_ids)
        self.n_buyers = len(self.buyer_ids)
        self.seller_ids = set(seller_ids)
        self.n_sellers = len(self.seller_ids)
        self.agent_ids = self.buyer_ids.union(self.seller_ids)
        self.max_steps = max_steps
        self.max_group_size = max(self.buyer_ids, self.n_sellers)
        self.max_n_deals = min(self.buyer_ids, self.n_sellers)
        self.reset()

    def reset(self):
        """Reset the market to its initial unmatched state."""
        self.time = 0
        self.done = set()
        self.offer_history = list()
        self.deal_history = list()

    def step(self, offers):
        """
        Compute the next market state given a set of offers.

        This function will update each of the class attributes ``time``,
        ``done``, ``offer_history`` and ``deal_history``.

        Parameters
        ----------
        offers: dict
            A dictionary of offers indexed by agent_id.

        Returns
        -------
        deals: dict
            A dictionary indexed by agent_id containing the deal price of each
            succesfully matched market agent.
        """
        bids = []
        asks = []

        for agent_id, offer in offers.items():
            if agent_id in self.done:
                continue
            elif agent_id in self.buyer_ids:
                bids.append((offer, agent_id))
            elif agent_id in self.seller_ids:
                asks.append((offer, agent_id))
            else:
                raise RuntimeError(f"Received offer from unkown agent {agent_id}")

        deals = self.match(bids, asks)

        self.deal_history.append(deals)
        self.offer_history.append((bids, asks))
        self.time += 1

        for agent_id in deals:
            self.done.add(agent_id)

        if (
            self.time >= self.max_steps
            or self.buyer_ids.issubset(self.done)
            or self.seller_ids.issubset(self.done)
        ):
            self.done = self.agent_ids

        return deals

    @staticmethod
    def match(bids, asks):
        """
        Core matching algorithm for market engine.

        Matching is done as follows: If a bid is higher than an ask, then the
        buyer will be matched with the seller. If there are multiple bids and
        asks that could be matched, then matching is determined by the offer
        placed: the highest bidder will be matched with the lowest asking
        seller, the second highest bidder will be matched with the second
        lowest seller and so on.  No match will happen if none of the bids
        exceed the asks.

        The deal price is determined to be the mid-price of the matched bid and
        ask. Note: the deal price does not have to be monotone in the offers of
        either side, i.e., it could happen that the second highest bidder
        obtains a better deal price than the highest bidder.

        Parameters
        ----------
        bids: list of tuples
            A list of the form ``[(bid1, agent_id1), (bid2, agent_id2), ...]``.
        asks: list of tuples
            A list of the form ``[(ask1, agent_id1), (ask2, agent_id2), ...]``.

        Returns
        -------
        deals: dict
            A dictionary indexed by agent_id containing the deal price. Only
            agents that were matched will be in this dict. Note: the same deal
            price will appear twice under the buyer and seller id.
        """
        bids.sort(reverse=True)
        asks.sort(reverse=False)
        deals = dict()
        n = min(len(bids), len(asks))
        for (bid, buyer_id), (ask, seller_id) in zip(bids[0:n], asks[0:n]):
            if bid >= ask:
                price = (bid + ask) / 2
                deals[buyer_id] = price
                deals[seller_id] = price
            else:
                break
        return deals

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

        # getting the realized prices in the original ordering for buyers&sellers, no deal means 0
        deals_buyers_orginal = sorted_deal_prices[:, : self.n_buyers].gather(
            1, b_actions_indices.argsort(1)
        )

        deals_sellers_orginal = sorted_deal_prices[:, : self.n_sellers].gather(
            1, s_actions_indices.argsort(1)
        )
        return deals_sellers_orginal, deals_buyers_orginal