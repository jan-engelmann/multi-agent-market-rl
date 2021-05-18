import itertools
import numpy as np

import torch

from tianshou.data import Batch, ReplayBuffer
from tqdm import tqdm

from marl_env.loss_setting import SimpleLossSetting


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
            obs, act, rew, _, _ = self.env.step()
            s_rew, b_rew = torch.split(rew, [self.env.n_sellers, self.env.n_buyers], dim=1)
            tot_loss = self.loss.get_losses(self.env, s_rew, b_rew)
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


class DeepQTrainer:
    torch.autograd.set_detect_anomaly(True)

    def __init__(self, env, memory_size, replay_start_size):
        self.env = env
        self.env.reset()

        self.buffer = self.set_replay_buffer(memory_size, replay_start_size)

    def set_replay_buffer(self, memory_size, replay_start_size):
        """
        Initializes the first N replay buffer entries with random actions.
        Parameters
        ----------
        memory_size: int
            Total size of the replay buffer
        replay_start_size: int
            Number of buffer entries to be initialized with random actions (corresponds to the integer N)

        Returns
        -------
        buffer: tianshou.data.ReplayBuffer
            Initialised replay buffer
        """
        buffer = ReplayBuffer(size=memory_size)

        for step in range(replay_start_size):
            obs, act, rew, obs_next, done = self.env.step(random_action=True)
            history_batch = Batch(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next)
            buffer.add(history_batch)
        return buffer

    def train(self, n_episodes, batch_size):
        for eps in range(n_episodes):
            self.env.reset()
            while not self.env.done:
                obs, act, rew, obs_next, done = self.env.step()
                history_batch = Batch(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next)
                self.buffer.add(history_batch)

                # Sample a random minibatch of transitions from the replay buffer
                batch_data, indice = self.buffer.sample(batch_size=batch_size)
                break






