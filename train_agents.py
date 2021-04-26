from marl_env.info_setting import BlackBoxSetting, OfferInformationSetting
from marl_env.markets import MarketMatchHiLo
from marl_env.agents import DummyAgent
from marl_env.environment import MultiAgentEnvironment
from marl_env.agents import GymRLAgent
from gym.spaces import Discrete

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
    n_proc = 4

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

    def select_policy(agent_id):
        """This function maps the agent id to the policy id"""
        return agent_id

    with mp.Pool(n_proc) as pool:
        env = MultiAgentEnvironment(
            sellers,
            buyers,
            market,
            info_setting,
            n_environments,
            pool,
        )
        my_policy = (None, env.observation_space, Discrete(20), {})
        # None means this policy needs to be learned.
        # Note: The action space should match that of the GymRLAgent defined earlier

        # We name our policies the same as our RL agents
        policies = {
            "S1": my_policy,
            "B1": my_policy,
            "S2": my_policy,
            "B2": my_policy,
        }

        from ray.rllib.agents import dqn
        from ray.rllib.env.multi_agent_env import MultiAgentEnv


        class MultiWrapper(MultiAgentTrainingEnv, MultiAgentEnv):
            def __init__(self, env_config):
                super().__init__(**env_config)

        trainer = dqn.DQNTrainer(env=MultiWrapper, config={
            "env_config": {"rl_agents": sellers + buyers, "setting": info_setting},
            "timesteps_per_iteration": 30,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": select_policy
            },
            "log_level": "ERROR",
        })

        trainer.train()
