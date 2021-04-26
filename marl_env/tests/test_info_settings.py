import pytest

from marl_env.info_setting import (
    BlackBoxSetting,
    OfferInformationSetting,
    TimeInformationWrapper,
)
from marl_env.markets import MarketMatchHiLo

# TODO: @Ben write tests that check actual functionality not only dimensions


def test_black_box():
    n_sellers = 10
    n_buyers = 10
    n_agents = n_sellers + n_buyers
    n_environments = 15
    market = MarketMatchHiLo(
        list(range(n_sellers)), list(range(n_buyers)), n_environments=n_environments
    )
    info_setting = BlackBoxSetting()
    res = info_setting.get_states(market)
    assert res.shape == (n_agents, n_environments, 1)


def test_offer_information_setting():
    n_sellers = 10
    n_buyers = 10
    n_agents = n_sellers + n_buyers
    n_environments = 15
    n_offers = 5
    market = MarketMatchHiLo(
        list(range(n_sellers)), list(range(n_buyers)), n_environments=n_environments
    )
    info_setting = OfferInformationSetting(n_offers=n_offers)
    res = info_setting.get_states(market)
    assert res.shape == (n_agents, n_environments, 2 * n_offers)


def test_time_information_wrapper():
    n_sellers = 10
    n_buyers = 10
    n_agents = n_sellers + n_buyers
    n_environments = 15
    n_offers = 5
    market = MarketMatchHiLo(
        list(range(n_sellers)), list(range(n_buyers)), n_environments=n_environments
    )
    info_setting = TimeInformationWrapper(OfferInformationSetting(n_offers=n_offers))
    res = info_setting.get_states(market)
    assert res.shape == (n_agents, n_environments, 2 * n_offers + 1)
