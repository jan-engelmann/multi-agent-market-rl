from marl_env.info_setting import BlackBoxSetting, OfferInformationSetting
from marl_env.markets import MarketMatchHiLo
from marl_env.agents import DummyAgent
from marl_env.environment import MultiAgentEnvironment
from marl_env.agents import GymRLAgent

import torch
import torch.multiprocessing as mp
import pandas as pd

from tqdm import tqdm
import time

if __name__ == "__main__":
    n_sellers = 50
    n_buyers = 50
    n_environments = 15
    n_processes = [4]  # [1, 2, 4, 6, 8]
    times = []
    n_iterations = 1
    n_steps = 50

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

    sellers = [
        GymRLAgent("seller", 80, "S1", max_factor=0.25, discretization=20),
        GymRLAgent("seller", 90, "S2", max_factor=0.25, discretization=20),
    ]
    buyers = [
        GymRLAgent("buyer", 120, "B1", max_factor=0.25, discretization=20),
        GymRLAgent("buyer", 110, "B2", max_factor=0.25, discretization=20),
    ]

    my_policy = (None, env.observation_space, Discrete(20), {})
    # None means this policy needs to be learned.
    # Note: The action space should match that of the GymRLAgent defined earlier

    def select_policy(agent_id):
        """This function maps the agent id to the policy id"""
        return agent_id

    # We name our policies the same as our RL agents
    policies = {
        "S1": my_policy,
        "B1": my_policy,
        "S2": my_policy,
        "B2": my_policy,
    }

    for n_proc in tqdm(n_processes, desc="n_processes"):
        for it in tqdm(range(n_iterations), desc="n_iterations", leave=False):
            duration = 0.0
            start = time.time()
            with mp.Pool(n_proc) as pool:
                env = MultiAgentEnvironment(
                    sellers,
                    buyers,
                    market,
                    info_setting,
                    n_environments,
                    pool,
                )
                for i in range(n_steps):
                    env.step()
            end = time.time()
            duration += end - start
        times.append(duration / n_iterations)  # type: ignore

    df = pd.DataFrame(dict(n_processes=n_processes, time=times))
    print(df)
    print("Finished")