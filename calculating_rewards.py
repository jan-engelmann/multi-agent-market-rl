import torch
import random
import numpy as np
import time

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def calculate_rewards(s_actions, s_reservations, b_actions, b_reservations):
    n_environments = s_actions.shape[0]
    n_sellers = s_actions.shape[1]
    n_buyers = b_actions.shape[1]

    # sort actions of sellers and buyers
    # using mechanism that highest buying offer is matched with lowest selling offer
    s_actions_sorted, s_actions_indices = s_actions.sort()
    b_actions_sorted, b_actions_indices = b_actions.sort(descending=True)

    max_n_deals = min(n_buyers, n_sellers)
    max_group_size = max(n_buyers, n_sellers)

    # get mask for all deals that happen
    D = b_actions_sorted[:, :max_n_deals] - s_actions_sorted[:, :max_n_deals]
    D = torch.cat((D, torch.zeros(n_environments, max_group_size - max_n_deals)), dim=1)
    deal_mask = D > 0

    s_realized_sorted_deals = s_actions_sorted.clone()
    s_realized_sorted_deals[~deal_mask[:, : s_actions.shape[1]]] = 0

    b_realized_sorted_deals = b_actions_sorted.clone()
    b_realized_sorted_deals[~deal_mask[:, : b_actions.shape[1]]] = 0

    # calculating deal prices
    sorted_deal_prices = (
        b_realized_sorted_deals[:, :max_n_deals]
        + s_realized_sorted_deals[:, :max_n_deals]
    ) / 2.0

    # getting the realized prices in the original ordering for buyers&sellers, no deal means 0
    deals_buyers_orginal = torch.cat(
        (sorted_deal_prices, torch.zeros(n_environments, max(0, n_buyers - n_sellers))),
        dim=1,
    ).gather(1, b_actions_indices.argsort(1))

    deals_sellers_orginal = torch.cat(
        (sorted_deal_prices, torch.zeros(n_environments, max(0, n_sellers - n_buyers))),
        dim=1,
    ).gather(1, s_actions_indices.argsort(1))

    # calculating rewrds for sellers and buyers
    rewards_sellers = deals_sellers_orginal - s_reservations[None, :]
    rewards_buyers = b_reservations[None, :] - deals_buyers_orginal

    return rewards_sellers, rewards_buyers


if __name__ == "__main__":
    n_sellers = 4
    n_buyers = 2
    n_environments = 3

    s_actions = torch.rand(n_environments, n_sellers)
    b_actions = torch.rand(n_environments, n_buyers)
    s_reservations = torch.rand(n_sellers)
    b_reservations = torch.rand(n_buyers)

    rewards_sellers, rewards_buyers = calculate_rewards(
        s_actions, s_reservations, b_actions, b_reservations
    )

    print(rewards_sellers)
    print(rewards_buyers)