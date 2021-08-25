import torch


class NetworkSetting:
    """
    Abstract network setting class.
    Used to define your custom neural networks which can then be used by intelligent agents
    """
    def __init__(self, in_features, out_features, device=torch.device('cpu'), **kwargs):
        """
        Initialises in_features and out_features needed to construct a fitting neural network

        Parameters
        ----------
        in_features: int
            Determined by the info_setting --> How many features does the agent see
        out_features: int
            Determined by the action space --> Number of legal actions including No action
        device: torch.device, optional (default cpu)
            Device on which the neural network is intended to run (cpu or gpu)
        kwargs: Optional additional keyword arguments
            load_weights_path: str, optional (default=False)
                If a path is provided, agent will try to load pretrained weights from there.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.load_weights_path = kwargs.pop("load_weights_path", False)

    def define_network(self):
        """

        Returns
        -------
        network: torch.nn
            The wanted neural network
        """
        raise NotImplementedError

    def get_network(self):
        """
        Will return the wanted neural network model located on the intended device (cpu or gpu)

        Returns
        -------
        network: torch.nn
            The wanted neural network on the correct device
        """
        network = self.define_network()

        if self.load_weights_path:
            print(f"-- Loading model weights from {self.load_weights_path}")
            network = torch.load(self.load_weights_path)
        network = network.to(self.device)

        return network


class SimpleExampleNetwork(NetworkSetting):
    """
    A simple network fulfilling the role of being an example
    """
    def __init__(self, in_features, out_features, device=torch.device('cpu'), **kwargs):
        super(SimpleExampleNetwork, self).__init__(in_features,
                                                   out_features,
                                                   device=device,
                                                   **kwargs)

    def define_network(self):
        """
        Defines a simple network for the purpose of being an example

        Returns
        -------
        network: torch.nn
            The wanted neural network
        """
        network = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.out_features),
        )

        return network
