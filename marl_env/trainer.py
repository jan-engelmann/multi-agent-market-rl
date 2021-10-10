import itertools

import torch

from collections import deque
from tianshou.data import Batch
from replay_buffer import ReplayBuffer


class DeepQTrainer:
    torch.autograd.set_detect_anomaly(True)

    def __init__(
        self,
        env,
        memory_size,
        replay_start_size,
        **kwargs,
    ):
        """

        Parameters
        ----------
        env: Environment object
            The current environment class object.
        memory_size: int
            ReplayBuffer size
        replay_start_size: int
            Number of ReplayBuffer slots to be initialised with a uniform random policy before learning starts
        kwargs: Optional keyword arguments
            discount: float, optional (default=0.99)
                Multiplicative discount factor for Q-learning update
            update_frq: int, optional (default=100)
                Frequency (measured in episode counts) with which the target network is updated
            max_loss_history: int, optional (default=None)
                Number of previous episodes for which the loss will be saved for monitoring
                None --> All episode losses are saved
            max_reward_history: int, optional (default=None)
                Number of previous episodes for which the rewards will be saved for monitoring
                None --> All episode rewards are saved
            max_action_history: int, optional (default=None)
                Number of previous episodes for which the actions will be saved for monitoring
                None --> All episode actions are saved
            loss_min: int, optional (default=-5)
                Lower-bound for the loss to be clamped to
            loss_max: int, optional (default=5)
                Upper-bound for the loss to be clamped to
            save_weights: bool, optional (default=False)
                If true, all agent weights will be saved to the respective directory specified by the agent in question
        """
        self.env = env
        self.discount = kwargs.pop('discount', 0.99)
        self.update_frq = kwargs.pop('update_frq', 100)
        max_loss_history = kwargs.pop('max_loss_history', None)
        max_reward_history = kwargs.pop('max_reward_history', None)
        max_action_history = kwargs.pop('max_action_history', None)
        self.clamp_min = kwargs.pop('loss_min', -5)
        self.clamp_max = kwargs.pop('loss_max', 5)
        self.avg_loss_history = deque(maxlen=max_loss_history)
        self.avg_reward_history = deque(maxlen=max_reward_history)
        self.last_actions = deque(maxlen=max_action_history)

        self.save_weights = kwargs.pop('save_weights', False)

        assert self.clamp_min < self.clamp_max, "loss_min must be strictly smaller then loss_max"

        self.env.reset()
        self.buffer = self.set_replay_buffer(memory_size, replay_start_size)

    @staticmethod
    def get_agent_Q_target(agent, observations, agent_state):
        """
        Returns all the Q-targets of the different agents

        Parameters
        ----------
        agent: agent class instance
        observations: torch.tensor
            All current observations
        agent_state: torch.tensor
            The state (active/finished) of all agents

        Returns
        -------
        targets: torch.tensor
            Q-targets of all agents
        """
        target = agent.get_target(observations, agent_state=agent_state)
        return target

    @staticmethod
    def get_agent_Q_values(agent, observations, actions=None):
        """
        Returns all the Q-values of the different agents

        Parameters
        ----------
        agent: agent class instance
        observations: torch.tensor
            All current observations
        actions: torch.tensor
            All current actions

        Returns
        -------
        q_values: torch.tensor
            Q-values of all agents
        """
        q_values = agent.get_q_value(observations, actions=actions)
        return q_values

    def mse_loss(self, q_targets, q_values):
        """
        Custom MSE-loss with clamping

        Parameters
        ----------
        q_targets: torch.tensor
            All the Q-value targets
        q_values: torch.tensor
            All the Q-values chosen by the agents

        Returns
        -------
        loss: torch.tensor
            The clamped mean squared error loss
        """
        y_target = q_targets.mean(dim=0)
        prediction = q_values.mean(dim=0)

        loss = torch.clamp(torch.sub(y_target, prediction), self.clamp_min, self.clamp_max).square()
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
            obs, act, rew, _, obs_next, a_states, done = self.env.step(random_action=True)
            history_batch = Batch(
                obs=obs,
                act=act,
                rew=rew,
                done=done,
                obs_next=obs_next,
                a_states=a_states,
            )
            buffer.add(history_batch)
            self.env.reset()
        return buffer

    def generate_Q_targets(
        self, obs_next, reward, agent_state, end_of_eps, discount=0.99
    ):
        """
        Generates the Q-value targets used to update the agent QNetwork
        Parameters
        ----------
        obs_next: torch.Tensor
            Tensor containing all observations for sampled time steps t_j + 1
        reward: torch.Tensor
            Tensor containing all rewards for the sampled time steps t_j
        agent_state: torch.Tensor
            Tensor indicating if an individual agent was finished at sampled time steps t_j
        end_of_eps: ndarray
            Bool array indicating if episode was finished at sampled time steps t_j
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
            (agent, obs_batch.transpose(0, 1), agent_state_batch.unsqueeze(1))
            for agent, obs_batch, agent_state_batch in zip(
                self.env.all_agents,
                obs_next.squeeze().transpose(0, 1).unsqueeze(1),
                agent_state.squeeze().transpose(0, 1),
            )
        ]
        targets = torch.stack(
            list(
                itertools.starmap(
                    DeepQTrainer.get_agent_Q_target, agent_obs_next_tuples
                )
            ),
            dim=1,
        ).squeeze()
        targets = torch.mul(targets, discount)
        targets = torch.mul(
            targets, torch.tensor(~end_of_eps, device=targets.device).unsqueeze(0).transpose(0, 1)
        )
        targets = torch.add(targets, reward.squeeze().to(targets.device))
        return targets

    def generate_Q_values(self, obs, act):
        """
        Generates the Q-values for each agent

        Parameters
        ----------
        obs: torch.tensor
            All current observations
        act: torch.tensor
            All current actions

        Returns
        -------
        q_values: torch.tensor
            The Q-values of the individual agents
        """
        agent_obs_act_tuples = [
            (agent, obs_batch.transpose(0, 1), act_batch.transpose(0, 1))
            for agent, obs_batch, act_batch in zip(
                self.env.all_agents,
                obs.squeeze().transpose(0, 1).unsqueeze(1),
                act.squeeze().transpose(0, 1).unsqueeze(1),
            )
        ]
        q_values = torch.stack(
            list(
                itertools.starmap(DeepQTrainer.get_agent_Q_values, agent_obs_act_tuples)
            ),
            dim=1,
        ).squeeze()
        return q_values

    def train(self, n_episodes, batch_size):
        """
        Training method.

        Parameters
        ----------
        n_episodes: int
            Number of episodes (games) to train for
        batch_size: int
            Batch size used to update the agents network weights

        Returns
        -------
        list
            list[0]: Average loss history
            list[1]: Average reward history
            list[2]: Action history
        """
        for eps in range(n_episodes):
            self.env.reset()
            eps_loss = deque(maxlen=self.env.market.max_steps)
            eps_rew = deque(maxlen=self.env.market.max_steps)
            while not self.env.done:
                obs, act, rew, obs_next, a_states, done = self.env.step()
                history_batch = Batch(
                    obs=obs,
                    act=act,
                    rew=rew,
                    done=done,
                    obs_next=obs_next,
                    a_states=a_states,
                )
                self.buffer.add(history_batch)
                self.last_actions.append(act)

                # Sample a random minibatch of transitions from the replay buffer
                batch_data, indice = self.buffer.sample(batch_size=batch_size)

                q_targets = self.generate_Q_targets(
                    batch_data["obs_next"],
                    batch_data["rew"],
                    batch_data["a_states"],
                    batch_data["done"],
                    discount=self.discount,
                ).detach()
                q_values = self.generate_Q_values(batch_data["obs"], batch_data["act"])
                loss = self.mse_loss(q_targets, q_values)
                loss.backward(torch.ones((self.env.n_agents,), device=loss.device))

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
                eps_loss.append(loss.detach().to(torch.device('cpu')))
                eps_rew.append(rew.detach().to(torch.device('cpu')))
            avg_loss = torch.stack(list(eps_loss), dim=0).mean(dim=0)
            avg_rew = torch.stack(list(eps_rew), dim=0).mean(dim=0)
            self.avg_loss_history.append(avg_loss)
            self.avg_reward_history.append(avg_rew)
            if eps % self.update_frq == 0:
                # print("Updating target Network")
                for agent in self.env.all_agents:
                    agent.reset_target_network()
        if self.save_weights:
            print("Saving model weights")
            for agent in self.env.all_agents:
                agent.save_model_weights()
            print("")
        return (
            list(self.avg_loss_history),
            list(self.avg_reward_history),
            list(self.last_actions),
        )
