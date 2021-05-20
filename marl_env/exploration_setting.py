import numpy as np


class ExplorationSetting:
    """ """

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        """
        pass

    def update(self):
        pass


class LinearExplorationDecline(ExplorationSetting):
    """ """

    def __init__(self, **kwargs):
        self.epsilon = kwargs.pop("initial_expo", 1.0)
        self.tot_steps = kwargs.pop("n_expo_steps", 100000)
        self.current_step = 0

        self.all_epsilon_values = np.linspace(
            self.epsilon, kwargs.pop("final_expo", 0), num=self.tot_steps
        )

    def update(self):
        if self.current_step < self.tot_steps - 1:
            self.current_step += 1
            self.epsilon = self.all_epsilon_values[self.current_step]

    def reset(self):
        self.current_step = 0
