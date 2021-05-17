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

    def __init__(self, env, training_steps=5, learning_rate=0.5):
        self.env = env
        self.training_steps = training_steps
        self.learning_rate = learning_rate
        self.loss = SimpleLossSetting()
        self.env.reset()
        self.setup()

    def setup(self):
        self.optimizers = [
            torch.optim.Adam(agent.model.parameters(), lr=self.learning_rate) for agent in self.env.all_agents
        ]
        # Set all initial gradients to zero. Is this really needed?
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def train(self):
        loss_matrix = np.zeros(
            (self.env.n_environments, self.env.n_agents, self.training_steps)
        )
        for t_step in tqdm(range(self.training_steps)):
            obs, rew, actions = self.env.step()
            tot_loss = self.loss.get_losses(self.env, rew[0], rew[1])
            tot_loss.backward(torch.ones_like(tot_loss))

            for agent in range(self.env.n_agents):
                old_params = {}
                for name, param in enumerate(self.env.all_agents[agent].model.parameters()):
                    old_params[name] = param.clone()
                self.optimizers[agent].step()  # parameter update
                # if list(self.env.all_agents[0].model.parameters())[0].grad.data > 0:
                #     print("Step: ", t_step, " Agent: ", agent, " after step True")
                #     print(list(self.env.all_agents[0].model.parameters())[0].grad)
                # for name, param in enumerate(self.env.all_agents[agent].model.parameters()):
                #     if not (old_params[name] == param):
                #         print("Performed parameter update for step:", t_step, "and agent: ", agent)
                self.optimizers[agent].zero_grad()
                # print("Step: ", t_step, " Agent: ", agent, " after reset")
                # print(list(self.env.all_agents[0].model.parameters())[0].grad)

            loss_matrix[:, :, t_step] = tot_loss.detach().numpy()

        return loss_matrix
