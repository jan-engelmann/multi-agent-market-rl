import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


if __name__ == "__main__":
    N_S = 500000
    N_B = 300000
    N_E = 30000
    S = torch.rand(N_E, N_S)
    B = torch.rand(N_E, N_B)

    # S = torch.tensor()
    print("SELLERS")
    print(S)
    print("BUYERS")
    print(B)

    S_sorted, S_indices = S.sort()
    print("SELLERS SORTED")
    print(S_sorted)
    print("SELLER INDICES")
    print(S_indices)

    B_sorted, B_indices = B.sort(descending=True)
    print("BUYERS SORTED")
    print(B_sorted)
    print("BUYER INDICES")
    print(B_indices)

    max_n_deals = min(N_B, N_S)
    max_dim = max(N_B, N_S)

    D = B_sorted[:, :max_n_deals] - S_sorted[:, :max_n_deals]
    print("DEALS DEALS DEALS")
    D = torch.cat((D, torch.zeros(N_E, max_dim - max_n_deals)), dim=1)
    print(D)

    print("DIFFERENCE")
    print(D)

    deal_mask = D > 0
    print("DEAL MATRIX")
    print(deal_mask)

    print("SELLER DEALS")
    S_sorted_deals = S_sorted.clone()
    S_sorted_deals[~deal_mask[:, : S.shape[1]]] = 0
    print(S_sorted_deals)

    print("BUYER DEALS")
    B_sorted_deals = B_sorted.clone()
    B_sorted_deals[~deal_mask[:, : B.shape[1]]] = 0
    print(B_sorted_deals)

    print("DEAL PRICES")
    sorted_deal_prices = (
        B_sorted_deals[:, :max_n_deals] + S_sorted_deals[:, :max_n_deals]
    ) / 2.0
    print(sorted_deal_prices)

    print("DEALS BUYERS ORIGINAL")
    deals_buyers_orginal = sorted_deal_prices.gather(1, B_indices.argsort(1))
    print(deals_buyers_orginal)

    print("DEALS Sellers ORIGINAL")
    deals_sellers_orginal = torch.cat(
        (sorted_deal_prices, torch.zeros(N_E, max_dim - max_n_deals)), dim=1
    ).gather(1, S_indices.argsort(1))
    print(deals_sellers_orginal)

    print(
        torch.cat((sorted_deal_prices, torch.zeros(N_E, max_dim - max_n_deals)), dim=1)
    )

    # D_B = deal_mask.gather(1, B_indices.argsort(1))
    # print("D_B")
    # print(D_B)
    # test = torch.tensor([[]])

    # original = torch.tensor([[20, 22, 24, 21], [12, 14, 10, 11], [34, 31, 30, 32]])
    # sorted, index = original.sort()
    # unsorted = sorted.gather(1, index.argsort(1))
    # assert torch.all(original == unsorted)

    # print((original == 20).nonzero())

    # for agent_id, offer in offers.items():
    #     if agent_id in self.done:
    #         continue
    #     elif agent_id in self.buyers:
    #         bids.append((offer, agent_id))
    #     elif agent_id in self.sellers:
    #         asks.append((offer, agent_id))
    #     else:
    #         raise RuntimeError(f"Received offer from unkown agent {agent_id}")

    # for e in range(N_E):
    #     for ask, bid in zip(s_sorted[e, :], b_sorted[e, :]):
    #         if ask < bid:
    #             print(matched, bid, asked)
