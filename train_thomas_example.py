from marl_env.trainer import ThomasSimpleTrainer
from marl_env.environment import MultiAgentEnvironment
from marl_env.markets import MarketMatchHiLo
from marl_env.info_setting import OfferInformationSetting, TimeInformationWrapper
from marl_env.agents import LinearAgent

import numpy as np
import datetime
import torch

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    n_sellers = 10
    n_buyers = 8
    n_environments = 3
    info_setting = TimeInformationWrapper(OfferInformationSetting(), max_steps=90000000)

    sellers = [
        LinearAgent(info_setting.observation_space.shape[0]) for _ in range(n_sellers)
    ]
    buyers = [
        LinearAgent(info_setting.observation_space.shape[0]) for _ in range(n_buyers)
    ]

    market = MarketMatchHiLo(n_sellers, n_buyers, n_environments, max_steps=30)

    env = MultiAgentEnvironment(sellers, buyers, market, info_setting, n_environments)

    trainer = ThomasSimpleTrainer(env)
    res = trainer.train()
    np.savetxt(f"{datetime.datetime.now()}training_results.csv", res, delimiter=",")
