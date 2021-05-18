import torch
import numpy as np


class AgentSetting:
    """
    Abstract agent class

    """
    def __init__(self, role, reservation, in_features, action_boundary, q_lr=0.001, target_lr=0.001) -> None:
        pass

    def get_action(self, observation, epsilon=0.05):
        raise NotImplementedError

    def random_action(self, observation=None, epsilon=None):
        raise NotImplementedError


class DQNAgent(AgentSetting):
    def __init__(self, role, reservation, in_features, action_boundary, q_lr=0.001, target_lr=0.001) -> None:
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
        q_lr: float, optional (default=0.001)
            Learning rate provided to the Q-Network optimizer
        target_lr: float, optional (default=0.001)
            Learning rate provided to the Target-Network optimizer
        """
        assert role in ["buyer", "seller"], "role should be 'buyer' or 'seller'"
        assert reservation > 0, "reservation price needs to be larger then zero"
        self.role = role
        self.reservation = reservation

        # For Buyer: Action space is the closed interval of [min_s_reservation, self.reservation]
        #     Where min_s_reservation denotes the smallest reservation price of all sellers. Therefore this action will
        #     result in 'no action' since a buyer must always bide higher then the reservation price of a seller.
        #     The highest possible action is given by the reservation price of the buyer that denotes his budget.
        #
        # For Seller: Action space is the closed interval of [self.reservation, max_b_reservation]
        #     Where max_b_reservation denotes the largest reservation price of all sellers. Therefore this action will
        #     result in 'no action' since the asking price of a seller must be smaller then the reservation price of a
        #     buyer.
        #     The lowest possible action is given by the reservation price of a seller guaranteeing a minimal profit
        #     of 0.5 in case a deal is reached.
        if role == "buyer":
            self.action_space = np.arange(action_boundary, reservation + 1)
        else:
            self.action_space = np.arange(reservation, action_boundary + 1)

        self.qNetwork = None
        self.q_opt = None

        self.targetNetwork = None
        self.target_opt = None

        # Number of out_features is equal to the number of possible actions. Therefore the number of out_features is
        # given by the length of the action_space.
        out_features = len(self.action_space)
        self.set_q_network(in_features, out_features)
        self.set_target_network(in_features, out_features)
        self.reset_target_network()

        self.set_optimizers(q_lr, target_lr)

    def set_q_network(self, in_features, out_features):
        # in_features is determined by the info_setting --> How many features does the agent see.
        # out_features is determined by the discretisation_setting --> Number of legal actions including No action.
        self.qNetwork = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_features)
        )

    def set_target_network(self, in_features, out_features):
        # in_features is determined by the info_setting --> How many features does the agent see.
        # out_features is determined by the discretisation_setting --> Number of legal actions including No action.
        self.targetNetwork = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_features)
        )

    def reset_target_network(self):
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())

        if self.target_opt:
            state_dict = self.target_opt.state_dict()
            self.target_opt = torch.optim.Adam(self.targetNetwork.parameters(), lr=0.001)
            self.target_opt.load_state_dict(state_dict)
            self.target_opt.zero_grad()

    def set_optimizers(self, q_lr, target_lr):
        self.q_opt = torch.optim.Adam(self.qNetwork.parameters(), lr=q_lr)
        self.target_opt = torch.optim.Adam(self.targetNetwork.parameters(), lr=target_lr)

        self.q_opt.zero_grad()
        self.target_opt.zero_grad()

    def get_action(self, observation, epsilon=0.05):
        """
        Parameters
        ----------
        observation: torch.Tensor
        epsilon: float
            epsilon defines the exploration rate (0,1). With a probability of epsilon the agent will perform a random
            action.

        Returns
        -------
        action_price: int
        """
        if torch.bernoulli(torch.Tensor([epsilon])):
            idx = torch.randint(len(self.action_space), (1,))
        else:
            q_values = self.qNetwork(observation).squeeze(1)
            idx = torch.argmax(q_values)
        action_price = self.action_space[idx]
        return action_price

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

    def get_target(self, observation):
        q_values = self.targetNetwork(observation).squeeze(1)
        max_q, _ = torch.max(q_values)
        return torch.Tensor([max_q])
