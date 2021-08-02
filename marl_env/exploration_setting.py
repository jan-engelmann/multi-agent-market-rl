import torch


class ExplorationSetting:
    """
    Abstract exploration setting class
    """

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        """
        self.epsilon = 0.0

    def update(self):
        pass


class LinearExplorationDecline(ExplorationSetting):
    """

    Parameters
    ----------
    kwargs:
        initial_expo: float, optional (default=1.0)
            Initial exploration probability
        n_expo_steps: int, optional (default=100000)
            Number of time steps over which the exploration rate will decrease linearly
        final_expo: float, optional (default=0.0)
            Final exploration rate
    """

    def __init__(self, **kwargs):
        super(LinearExplorationDecline, self).__init__(**kwargs)

        self.epsilon = kwargs.pop("initial_expo", 1.0)
        self.tot_steps = kwargs.pop("n_expo_steps", 100000)
        self.current_step = 0

        self.all_epsilon_values = torch.linspace(
            self.epsilon, kwargs.pop("final_expo", 0), self.tot_steps
        )

    def update(self):
        if self.current_step < self.tot_steps - 1:
            self.current_step += 1
            self.epsilon = self.all_epsilon_values[self.current_step]

    def reset(self):
        self.current_step = 0
