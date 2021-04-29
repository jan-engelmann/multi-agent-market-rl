import itertools
import numpy as np

import torch

from tqdm import tqdm

from marl_env.loss_setting import SimpleLossSetting


class ThomasSimpleTrainer:
    def __init__(
        self, env, training_steps=1000, obs_collection_steps=50, discount_factor=0.99
    ):
        self.env = env
        self.training_steps = training_steps
        self.obs_collection_steps = obs_collection_steps
        self.discount_coefficients = (discount_factor) ** torch.arange(
            obs_collection_steps
        )
        self.env.reset()
        self.setup()

    def setup(self):
        self.optimizers = [
            torch.optim.Adam(agent.model.parameters()) for agent in self.env.all_agents
        ]

    def train(self):
        # rollout experience collection
        loss_matrix = np.zeros(
            (self.env.n_agents, self.training_steps, self.obs_collection_steps)
        )
        # TODO: understand this loop. Does this make sense?
        # TODO: make environment break when a game is finished
        for t_step in tqdm(range(self.training_steps)):
            all_actions = []
            all_rewards = []
            all_obs = []
            for o_step in range(self.obs_collection_steps):
                obs, rew, actions = self.env.step()
                all_actions.append(torch.cat(actions, dim=1))
                all_rewards.append(torch.cat(rew, dim=1))
                all_obs.append(obs)

            for agent in range(self.env.n_agents):
                for i, obs in enumerate(all_obs):
                    agent_action = self.env.all_agents[agent].get_action(obs[:, agent])

                    # here we use a simple loss, q-value and belmann equation could replace it...
                    loss = (
                        -torch.stack(all_rewards[i:])[:, :, agent].sum(-1)
                        * self.discount_coefficients[i:]
                    ).sum()  # discount factor goes here
                    loss.backward()  # gradient calculation
                    self.optimizers[agent].step()  # parameter update
                    loss_matrix[
                        agent, t_step, i
                    ] = (
                        loss.detach().numpy()
                    )  # we should see this dropping as we repeat the training loop.
        return loss_matrix


class MeanAbsErrorTrainer:
    torch.autograd.set_detect_anomaly(True)
    def __init__(self, env, training_steps=1000):
        self.env = env
        self.training_steps = training_steps
        self.loss = SimpleLossSetting()
        self.env.reset()
        self.setup()

    def setup(self):
        self.optimizers = [
            torch.optim.Adam(agent.model.parameters()) for agent in self.env.all_agents
        ]

    def train(self):
        loss_matrix = np.zeros(
            (self.env.n_environments, self.env.n_agents, self.training_steps)
        )
        for t_step in tqdm(range(self.training_steps)):

            obs, rew, actions = self.env.step()
            env = self.env
            s_reward = rew[0]
            b_reward = rew[1]
            tot_loss = self.loss.get_losses(env, s_reward, b_reward)

            for agent in range(self.env.n_agents):
                loss = tot_loss[:, agent]
                loss = loss.sum()
                loss.backward(retain_graph=True)
                self.optimizers[agent].step()  # parameter update
                loss_matrix[:, agent, t_step] = loss.detach().numpy()
                print(f"Agent {agent} updated")
        return loss_matrix
