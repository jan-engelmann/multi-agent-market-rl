import torch
import numpy as np


class AgentSetting:
    """
    Abstract agent class

    """

    def __init__(self, role, reservation, in_features, action_boundary) -> None:
        assert role in ["buyer", "seller"], "role should be 'buyer' or 'seller'"
        assert reservation > 0, "reservation price needs to be larger then zero"
        self.role = role
        self.reservation = reservation
        self.in_features = in_features

        # For Buyer: Action space is the closed interval of [min_s_reservation, self.reservation]
        #     Where min_s_reservation denotes the smallest reservation price of all sellers. Therefore this action will
        #     result in 'no action' since a buyer must always bide higher then the reservation price of a seller.
        #     The highest possible action is given by the reservation price of the buyer that denotes his budget.
        #
        # For Seller: Action space is the closed interval of [self.reservation, max_b_reservation]
        #     Where max_b_reservation denotes the largest reservation price of all buyers. Therefore this action will
        #     result in 'no action' since the asking price of a seller must be smaller then the reservation price of a
        #     buyer.
        #     The lowest possible action is given by the reservation price of a seller guaranteeing a minimal profit
        #     of 0.5 in case a deal is reached.
        if role == "buyer":
            self.action_space = np.arange(action_boundary, reservation + 1)
        else:
            self.action_space = np.arange(reservation, action_boundary + 1)

    def get_action(self, observation, epsilon=0.05):
        raise NotImplementedError

    def random_action(self, observation=None, epsilon=None):
        raise NotImplementedError


class DQNAgent(AgentSetting):
    def __init__(
        self, role, reservation, in_features, action_boundary, **kwargs
    ) -> None:
        """
        Agents are implemented in such a manner, that asking and bidding prices are given as integer values. Therefore
        the unit of the price will be equal to the smallest possible unit of currency e.g. for CHF 1 == 0.05 CHF

        Parameters
        ----------
        role: str
            role can be 'buyer' or 'seller'
        reservation: int
            The reservation price needs to be in the open interval of (0, infinity). In the case of a seller, the
            reservation price denotes the fixed costs and in the case of a buyer, the reservation price denotes the
            budget of the agent.
        in_features: int
            Number of features observed by the agent
        action_boundary: int
            In case the agent is a seller, action_boundary should equal the largest reservation price of all buyers.
            In case the agent is a buyer, action_boundary should equal the smallest reservation price of all sellers.

        kwargs:
            q_lr: float, optional (default=0.001)
                Learning rate provided to the Q-Network optimizer
        """
        super(DQNAgent, self).__init__(role, reservation, in_features, action_boundary)

        self.qNetwork = None
        self.targetNetwork = None
        self.q_opt = None

        # Number of out_features is equal to the number of possible actions. Therefore the number of out_features is
        # given by the length of the action_space.
        out_features = len(self.action_space)
        self.set_q_network(self.in_features, out_features)
        self.set_target_network(self.in_features, out_features)
        self.reset_target_network()
        self.set_optimizers(kwargs.pop("q_lr", 0.001))

    def set_q_network(self, in_features, out_features):
        # in_features is determined by the info_setting --> How many features does the agent see.
        # out_features is determined by the discretisation_setting --> Number of legal actions including No action.
        self.qNetwork = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_features),
        )

    def set_target_network(self, in_features, out_features):
        # in_features is determined by the info_setting --> How many features does the agent see.
        # out_features is determined by the discretisation_setting --> Number of legal actions including No action.
        self.targetNetwork = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_features),
        )

    def reset_target_network(self):
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())

    def set_optimizers(self, q_lr):
        self.q_opt = torch.optim.Adam(self.qNetwork.parameters(), lr=q_lr)
        self.q_opt.zero_grad()

    def get_q_value(self, observation, actions=None):
        """

        Parameters
        ----------
        observation: torch.Tensor
            Agent observations. Should have shape (batch_size, observation_size)
        actions: torch.Tensor, optional (default = False)
            Will provide the Q values corresponding to the provided actions.
            actions should have shape (batch_size, 1)
            If no actions are provided, the Q value will correspond to the maximal value
        Returns
        -------
        max_q: torch.Tensor
            Tensor containing all Q values. Has shape (batch_size, 1)
        """
        q_values = self.qNetwork(observation)
        if torch.is_tensor(actions):
            indices = torch.stack(
                [torch.tensor(self.action_space) == act for act in actions]
            )
            max_q = q_values[indices]
        else:
            max_q, _ = torch.max(q_values, dim=1)
        return max_q

    def get_action(self, observation, epsilon=0.05):
        """
        Parameters
        ----------
        observation: torch.Tensor
        epsilon: float
            epsilon defines the exploration rate [0,1]. With a probability of epsilon the agent will perform a random
            action.

        Returns
        -------
        action_price: int
        """
        if torch.bernoulli(torch.Tensor([epsilon])):
            idx = torch.randint(len(self.action_space), (1,))
        else:
            q_values = self.qNetwork(observation).squeeze()
            idx = torch.argmax(q_values)
        action_price = self.action_space[idx]
        return torch.Tensor([action_price])

    def random_action(self, observation=None, epsilon=None):
        """
        A uniform random policy intended to populate the replay memory before learning starts

        Parameters
        ----------
        observation: Dummy parameter copying get_action()
        epsilon: Dummy parameter copying get_action()

        Returns
        -------
        action_price: int
            Uniformly sampled action price.
        """
        idx = torch.randint(len(self.action_space), (1,))
        action_price = self.action_space[idx]
        return torch.Tensor([action_price])

    def get_target(self, observation, agent_state=None):
        """

        Parameters
        ----------
        observation
        agent_state

        Returns
        -------

        """
        q_values = self.targetNetwork(observation)
        max_q, _ = torch.max(q_values, dim=1)

        # Get Q-values corresponding to 'no action'
        if self.role == "buyer":
            no_act = q_values[:, 0]
        else:
            no_act = q_values[:, -1]

        if not torch.is_tensor(agent_state):
            agent_state = torch.full_like(max_q, False, dtype=torch.bool)
        else:
            agent_state = agent_state.transpose(0, 1).squeeze()

        # Mask samples where the agent was done since the previous round with the Q-value of 'no action'
        res = torch.mul(max_q, ~agent_state) + torch.mul(no_act, agent_state)

        return res.unsqueeze(0).transpose(0, 1)
