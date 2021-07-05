import os
import torch
import pandas as pd


class AgentSetting:
    """
    Abstract agent class

    """

    def __init__(self, role, reservation, in_features, action_boundary, device=torch.device('cpu')) -> None:
        assert role in ["buyer", "seller"], "role should be 'buyer' or 'seller'"
        assert reservation > 0, "reservation price needs to be larger then zero"
        self.role = role
        self.reservation = reservation
        self.in_features = in_features
        self.device = device

        # Print role and reservation price of the agent being initialised
        print(f"-- role: {role}")
        print(f"-- reservation: {reservation}")

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
            assert action_boundary < (reservation + 1), (
                "Reservation price of buyer is <= the smallest reservation "
                "price of all sellers. This results in an empty action space."
                "Are you sure you wanted to make a buyer?"
            )
            self.action_space = torch.arange(action_boundary, reservation + 1, device=self.device)
        else:
            assert reservation < (action_boundary + 1), (
                "Reservation price of seller is >= the largest reservation "
                "price of all buyers. This results in an empty action space."
                "Are you sure you wanted to make a seller?"
            )
            self.action_space = torch.arange(reservation, action_boundary + 1, device=self.device)

    def get_action(self, observation, epsilon=0.05):
        raise NotImplementedError

    def random_action(self, observation=None, epsilon=None):
        raise NotImplementedError

    def get_q_value(self, observation, actions=None):
        raise NotImplementedError

    def get_target(self, observation, agent_state=None):
        raise NotImplementedError

    def reset_target_network(self):
        raise NotImplementedError

    def save_model_weights(self):
        raise NotImplementedError

    def load_model_weights(self):
        raise NotImplementedError


class DQNAgent(AgentSetting):
    def __init__(
        self, role, reservation, in_features, action_boundary, device=torch.device('cpu'), **kwargs
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
            save_weights_directory: str, optional (default="../saved_agent_weights/default_path/{self.agent_name}/")
                Directory to where model weights will be saved to.
            save_weights_file: str, optional (default="default_test_file.pt")
                File name of the saved weights. Must be a .pt or .pth file
            load_weights_path: str, optional (default=False)
                If a path is provided, agent will try to load pretrained weights from there.
        """
        print("Initialising DQNAgent")
        super(DQNAgent, self).__init__(role, reservation, in_features, action_boundary, device=device)

        self.qNetwork = None
        self.targetNetwork = None
        self.q_opt = None
        self.agent_name = kwargs.pop("agent_name", "Undefined_DQNAgent")
        lr = kwargs.pop("lr", 0.001)
        default_directory = f"../saved_agent_weights/default_path/{self.agent_name}/"
        self.save_weights_directory = kwargs.pop("save_weights_directory", default_directory)
        self.save_weights_file = kwargs.pop("save_weights_file", "default_test_file.pt")
        self.load_weights_path = kwargs.pop("load_weights_path", False)

        # Number of out_features is equal to the number of possible actions. Therefore the number of out_features is
        # given by the length of the action_space.
        out_features = len(self.action_space)
        self.set_q_network(self.in_features, out_features)
        self.set_target_network(self.in_features, out_features)
        self.reset_target_network()
        self.set_optimizers(lr)

        # Print optional agent settings
        print(f"-- lr: {lr}")
        print(f"-- save_weights_directory: {self.save_weights_directory}")
        print(f"-- save_weights_file: {self.save_weights_file}")
        print("")

    def set_q_network(self, in_features, out_features):
        # in_features is determined by the info_setting --> How many features does the agent see.
        # out_features is determined by the action space --> Number of legal actions including No action.
        self.qNetwork = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_features),
        )
        if self.load_weights_path:
            print(f"-- Loading model weights from {self.load_weights_path} for agent {self.agent_name}")
            self.load_model_weights()
        self.qNetwork = self.qNetwork.to(self.device)

    def set_target_network(self, in_features, out_features):
        # in_features is determined by the info_setting --> How many features does the agent see.
        # out_features is determined by the action space --> Number of legal actions including No action.
        self.targetNetwork = torch.nn.Sequential(
            torch.nn.Linear(in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_features),
        )
        self.targetNetwork = self.targetNetwork.to(self.device)

    def reset_target_network(self):
        # Make sure that the targetNetwork is still on the correct device
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())

    def set_optimizers(self, q_lr):
        # Optimizer should live on the same device as qNetwork dies.
        self.q_opt = torch.optim.Adam(self.qNetwork.parameters(), lr=q_lr)
        self.q_opt.zero_grad()

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
            obs = observation.to(self.device)
            q_values = self.qNetwork(obs).squeeze()
            idx = torch.argmax(q_values).unsqueeze(0)
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
        return action_price

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
        obs = observation.to(self.device)
        acts = actions.to(self.device)
        q_values = self.qNetwork(obs).squeeze()
        if torch.is_tensor(acts):
            indices = torch.stack(
                [self.action_space == act for act in acts]
            )
            max_q = q_values[indices]
        else:
            max_q, _ = torch.max(q_values, dim=1)
        return max_q

    def get_target(self, observation, agent_state=None):
        """

        Parameters
        ----------
        observation
        agent_state

        Returns
        -------

        """
        obs = observation.to(self.device)
        q_values = self.targetNetwork(obs).squeeze()
        max_q, _ = torch.max(q_values, dim=1)

        # Get Q-values corresponding to 'no action'
        if self.role == "buyer":
            no_act = q_values[:, 0]
        else:
            no_act = q_values[:, -1]

        if not torch.is_tensor(agent_state):
            agent_state = torch.full_like(max_q, False, dtype=torch.bool)
        else:
            agent_state = agent_state.to(self.device)
            agent_state = agent_state.transpose(0, 1).squeeze()

        # Mask samples where the agent was done since the previous round with the Q-value of 'no action'
        res = torch.mul(max_q, ~agent_state) + torch.mul(no_act, agent_state)

        return res.unsqueeze(0).transpose(0, 1)

    def load_model_weights(self):
        self.qNetwork = torch.load(self.load_weights_path)

    def save_model_weights(self):
        os.makedirs(self.save_weights_directory, exist_ok=True)
        print(f"-- Saving model weights for {self.agent_name}")
        torch.save(self.qNetwork.state_dict(), os.path.join(self.save_weights_directory, self.save_weights_file))


class ConstAgent(AgentSetting):
    def __init__(self, role, reservation, in_features, action_boundary, device=torch.device('cpu'), **kwargs):
        print("Initialising ConstAgent")
        super(ConstAgent, self).__init__(
            role, reservation, in_features, action_boundary, device=device
        )

        self.const_price = kwargs.pop(
            "const_price", (reservation + action_boundary) // 2.0
        )
        self.agent_name = kwargs.pop("agent_name", "Undefined_ConstAgent")

        # Print optional agent settings
        print(f"-- const_price: {self.const_price}")

        assert self.const_price in self.action_space, (
            f"The chosen constant price {self.const_price} is not included "
            f"in the action space! Possible integer constant prices must "
            f"be in the interval "
            f"[{self.action_space.min()}, {self.action_space.max()}]"
        )
        self.const_price = torch.tensor([self.const_price], device=self.device)
        self.q_opt = self.Optimizer()
        print("")

    def get_action(self, observation, epsilon=0.05):
        """
        Returns the constant action price of the agent
        Returns
        -------
        torch.Tensor of shape (1,) containing the action price of the agent
        """
        return self.const_price

    def get_target(self, observation, agent_state=None):
        """
        Dummy function --> ConstAgent is a zero intelligence agent
        """
        return torch.zeros((observation.shape[0]), device=self.device).unsqueeze(1)

    def get_q_value(self, observation, actions=None):
        """
        Dummy function --> ConstAgent is a zero intelligence agent
        """
        return torch.zeros((observation.shape[0]), device=self.device)

    def reset_target_network(self):
        pass

    def load_model_weights(self):
        pass

    def save_model_weights(self):
        print(f"-- {self.agent_name} has no model weights that can be saved... passing")
        pass

    class Optimizer:
        """
        Dummy optimizer
        """

        def __init__(self):
            pass

        def step(self):
            pass

        def zero_grad(self):
            pass


class HumanReplayAgent(AgentSetting):
    def __init__(self, role, reservation, in_features, action_boundary, device=torch.device('cpu'), **kwargs):
        print("Initialising HumanReplayAgent")
        super(HumanReplayAgent, self).__init__(
            role, reservation, in_features, action_boundary, device=device
        )
        data_type = kwargs.pop("data_type", "new_data")
        treatment = kwargs.pop("treatment", "FullLimS")
        player_id = kwargs.pop("id", 954)
        self.agent_name = kwargs.pop("agent_name", "Undefined_HumanReplayAgent")

        # Make better...
        if data_type == "new_data":
            data_path = "../HumanReplayData/NewData/new_data.csv"
        else:
            data_path = "../HumanReplayData/NewData/old_data.csv"

        # Print optional agent settings
        print(f"-- data_path: {data_path}")
        print(f"-- treatment: {treatment}")
        print(f"-- id: {player_id}")

        data = pd.read_csv(data_path)
        data = data.dropna(axis=0, subset=["valuation"])

        # Assert that the correct reservation price is used in the market.
        tmp = data.loc[(data["treatment"] == treatment) & (data["id"] == player_id)][
            "valuation"
        ].tolist()
        assert tmp == len(tmp) * [reservation], (
            "Reservation price from data does not match with the reservation "
            "price from the agent dict."
        )
        # Assert that the agent has the correct role in the market.
        tmp = data.loc[(data["treatment"] == treatment) & (data["id"] == player_id)][
            "side"
        ].tolist()
        assert [x.lower() for x in tmp] == len(tmp) * [role], (
            "Role from data set does not match with the role from " "the agent dict"
        )

        self.action_list = data.loc[
            (data["treatment"] == treatment) & (data["id"] == player_id)
        ]["bid"].tolist()
        assert len(self.action_list) != 0, (
            "Action list is empty --> Probably the chosen HumanReplayAgent does not "
            "exist. Double check the chosen agent configurations: 'data_type':... "
            "'treatment':..., 'id':..."
        )

        self.q_opt = self.Optimizer()
        self.action = None
        self.step = 0
        print("")

    def get_action(self, observation, epsilon=0.05):
        """
        Returns the next bid from the data set for every market step. If all data steps have been used, we will restart
        from the beginning.

        Returns
        -------
        torch.Tensor of shape (1,) containing the bid price of the agent
        """
        idx = self.step

        self.action = self.action_list[idx]
        self.step = (self.step + 1) % len(self.action_list)

        return torch.tensor([self.action], device=self.device)

    def get_target(self, observation, agent_state=None):
        """
        Dummy function --> HumanReplayAgent is a zero intelligence agent
        """
        return torch.zeros((observation.shape[0]), device=self.device).unsqueeze(1)

    def get_q_value(self, observation, actions=None):
        """
        Dummy function --> HumanReplayAgent is a zero intelligence agent
        """
        return torch.zeros((observation.shape[0]), device=self.device)

    def reset_target_network(self):
        pass

    def load_model_weights(self):
        pass

    def save_model_weights(self):
        print(f"-- {self.agent_name} has no model weights that can be saved... passing")
        pass

    class Optimizer:
        """
        Dummy optimizer
        """

        def __init__(self):
            pass

        def step(self):
            pass

        def zero_grad(self):
            pass
