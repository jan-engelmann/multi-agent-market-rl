import torch

from marl_env.agents import DummyAgent
from marl_env.markets import MarketMatchHiLo
from marl_env.info_setting import BlackBoxSetting, OfferInformationSetting
from marl_env.environment import MultiAgentEnvironment
import torch.multiprocessing as mp


def test_create_env_offer_info():
    n_sellers = 50
    n_buyers = 50
    n_environments = 15
    n_proc = 3

    sellers = [
        DummyAgent(idx, num)
        for idx, num in enumerate(torch.rand(n_sellers, n_environments))
    ]
    buyers = [
        DummyAgent(idx, num)
        for idx, num in enumerate(torch.rand(n_buyers, n_environments))
    ]
    buyer_ids = [agent.id for agent in buyers]
    seller_ids = [agent.id for agent in sellers]
    market = MarketMatchHiLo(buyer_ids, seller_ids, n_environments, max_steps=30)
    info_setting = OfferInformationSetting(n_offers=3)
    with mp.Pool(n_proc) as pool:
        env = MultiAgentEnvironment(
            sellers,
            buyers,
            market,
            info_setting,
            n_environments,
            pool,
        )
        env.step()
    return env


def test_create_env_black_box():
    n_sellers = 50
    n_buyers = 50
    n_environments = 15
    n_proc = 3

    sellers = [
        DummyAgent(idx, num)
        for idx, num in enumerate(torch.rand(n_sellers, n_environments))
    ]
    buyers = [
        DummyAgent(idx, num)
        for idx, num in enumerate(torch.rand(n_buyers, n_environments))
    ]
    buyer_ids = [agent.id for agent in buyers]
    seller_ids = [agent.id for agent in sellers]
    market = MarketMatchHiLo(buyer_ids, seller_ids, n_environments, max_steps=30)
    info_setting = BlackBoxSetting()
    with mp.Pool(n_proc) as pool:
        env = MultiAgentEnvironment(
            sellers,
            buyers,
            market,
            info_setting,
            n_environments,
            pool,
        )
        for i in range(20):
            env.step()
    return env
