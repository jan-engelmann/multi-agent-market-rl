import numpy as np


class ExplorationSetting:
    """

    """
    def __init__(self, initial_exp, final_exp, n_exp_steps):
        """

        Parameters
        ----------
        initial_exp
        final_exp
        n_exp_steps
        """
        pass

    def update(self):
        pass


class LinearExplorationDecline(ExplorationSetting):
    """

    """
    def __init__(self, initial_exp, final_exp, n_exp_steps):
        self.epsilon = initial_exp
        self.tot_steps = n_exp_steps
        self.current_step = 0

        self.all_epsilon_values = np.linspace(initial_exp, final_exp, num=n_exp_steps)

    def update(self):
        if self.current_step < self.tot_steps - 1:
            self.current_step += 1
            self.epsilon = self.all_epsilon_values[self.current_step]
