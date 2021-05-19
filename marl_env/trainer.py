import itertools
import numpy as np

import torch

from tqdm import tqdm
from collections import deque
from tianshou.data import Batch
from marl_env.replay_buffer import ReplayBuffer

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

    def __init__(self,
                 env,
                 memory_size,
                 replay_start_size,
                 update_frq=100,
                 discount=0.99,
                 max_loss_history=None,
                 max_reward_history=None):
        self.discount = discount
        self.update_frq = update_frq
        self.avg_loss_history = deque(maxlen=max_loss_history)
        self.avg_reward_history = deque(maxlen=max_reward_history)
        self.env = env
        self.env.reset()

        self.buffer = self.set_replay_buffer(memory_size, replay_start_size)

    @staticmethod
    def get_agent_Q_target(agent, observations):
        target = agent.get_target(observations)
        return target

    @staticmethod
    def get_agent_Q_values(agent, observations, actions=None):
        q_values = agent.get_q_value(observations, actions=actions)
        return q_values

    @staticmethod
    def mse_loss(q_targets, q_values):
        y_target = q_targets.mean(dim=0)
        prediction = q_values.mean(dim=0)

        loss = torch.clamp(torch.sub(y_target, prediction), -1.0, 1.0).square()
        return loss

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

    def generate_Q_targets(self, obs_next, reward, end_of_eps, discount=0.99):
        """

        Parameters
        ----------
        obs_next: torch.Tensor
            Tensor containing all observations for sampled time steps t_j + 1
        reward: torch.Tensor
            Tensor containing all rewards for the sampled time steps t_j
        end_of_eps: ndarray
            Bool array indicating if episode was finished at sampled time step t_j
        discount: float
            Discount factor gamma used in the Q-learning update.

        Returns
        -------
        targets: torch.Tensor
            Tensor containing all targets y_j used to perform a gradient descent step on (y_j - Q(s_j, a_j; theta))**2
            Tensor will have shape (batch_size, n_agents)
            targets[:, :n_sellers] contains all targets for the seller agents
            targets[:, n_sellers:] contains all targets for the buyer agents
        """
        agent_obs_next_tuples = [
            (agent, obs_batch.transpose(0, 1))
            for agent, obs_batch in zip(
                self.env.all_agents, obs_next.squeeze().transpose(0, 1).unsqueeze(1)
            )
        ]
        targets = torch.stack(
            list(
                itertools.starmap(
                    DeepQTrainer.get_agent_Q_target,
                    agent_obs_next_tuples
                )
            ),
            dim=1
        ).squeeze()
        targets = torch.mul(targets, discount)
        targets = torch.mul(targets, torch.Tensor(~end_of_eps).unsqueeze(0).transpose(0, 1))
        targets = torch.add(targets, reward.squeeze())
        return targets

    def generate_Q_values(self, obs, act):
        agent_obs_act_tuples = [
            (agent, obs_batch.transpose(0, 1), act_batch.transpose(0, 1))
            for agent, obs_batch, act_batch in zip(
                self.env.all_agents,
                obs.squeeze().transpose(0, 1).unsqueeze(1),
                act.squeeze().transpose(0, 1).unsqueeze(1)
            )
        ]
        q_values = torch.stack(
            list(
                itertools.starmap(
                    DeepQTrainer.get_agent_Q_values,
                    agent_obs_act_tuples
                )
            ),
            dim=1
        ).squeeze()
        return q_values

    def train(self, n_episodes, batch_size):
        for eps in range(n_episodes):
            self.env.reset()
            eps_loss = deque(maxlen=self.env.market.max_steps)
            eps_rew = deque(maxlen=self.env.market.max_steps)
            while not self.env.done:
                obs, act, rew, obs_next, done = self.env.step()
                history_batch = Batch(obs=obs, act=act, rew=rew, done=done, obs_next=obs_next)
                self.buffer.add(history_batch)

                # Sample a random minibatch of transitions from the replay buffer
                batch_data, indice = self.buffer.sample(batch_size=batch_size)

                q_targets = self.generate_Q_targets(batch_data['obs_next'],
                                                    batch_data['rew'],
                                                    batch_data['done'],
                                                    discount=self.discount).detach()
                q_values = self.generate_Q_values(batch_data['obs'], batch_data['act'])
                loss = DeepQTrainer.mse_loss(q_targets, q_values)
                loss.backward(torch.ones(self.env.n_agents))

                for agent in self.env.all_agents:
                    # old_params = {}
                    # for name, param in enumerate(agent.qNetwork.parameters()):
                    #     old_params[name] = param.clone()
                    agent.q_opt.step()
                    agent.q_opt.zero_grad()
                    # for name, param in enumerate(agent.qNetwork.parameters()):
                    #     bool_vec = (old_params[name] == param)
                    #     if False in bool_vec:
                    #         print("Performed parameter update for agent: ", agent)
                    #         print("Old values: ", old_params[name][~bool_vec], " New values: ", param[~bool_vec])

                # Monitoring features
                eps_loss.append(loss)
                eps_rew.append(rew)
            avg_loss = torch.stack(list(eps_loss), dim=0).mean(dim=0)
            avg_rew = torch.stack(list(eps_rew), dim=0).mean(dim=0)
            self.avg_loss_history.append(avg_loss)
            self.avg_reward_history.append(avg_rew)
            if eps % self.update_frq == 0:
                # print("Updating target Network")
                for agent in self.env.all_agents:
                    agent.reset_target_network()
        return self.avg_loss_history, self.avg_reward_history
