import torch


class LossSetting:
    """
    Abstract loss setting class.

    Attributes
    ----------
    """

    def __init__(self):
        pass

    def get_losses(self, env):
        """
        Compute the loss of all agents given the environment object.

        Parameters
        ----------
        env: MultiAgentEnvironment object
            The current environment object.

        Returns
        -------
        losses: torch.Tensor
            A Tensor of shape (n_environments, n_agents) containing the loss for each agent in every environment.
        """
        pass

    def get_loss(self, env):
        return self.get_losses(env)


class SimpleLossSetting(LossSetting):
    """
    Parameters
    ----------
    epsilon: float, optional (default=1e-6)
        Float value describing the smallest allowed difference between |aks - bid|.
        This is used to compute the maximum reward.

    Returns
    -------
    total_loss: torch.Tensor
        Tensor of shape (n_environments, n_agents) containing all losses.
        total_loss[:, :n_sellers] contains all losses from agents with the roll as seller
        total_loss[:, n_sellers:] contains all losses from agents with the roll as buyer
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def get_losses(self, env):
        self.n_sellers = env.n_sellers
        self.n_buyers = env.n_buyers
        self.n_environments = env.n_environments
        self.s_reservations = env.s_reservations.unsqueeze_(0).expand(self.n_environments, self.n_sellers)
        self.b_reservations = env.b_reservations.unsqueeze_(0).expand(self.n_environments, self.n_buyers)
        self.done_sellers = env.done_sellers
        self.done_buyers = env.done_buyers

        # Expand reservations to all environments. We want to do this, since we would like to mask agents no longer
        # participating. Because they might change the max reward possible.
        # TODO: Fix issue with masking values failing when used during backpropagation
        self.s_reservations[self.done_sellers] = self.s_reservations.max()
        self.b_reservations[self.done_buyers] = self.b_reservations.min()

        # Compute max reward for each agent and expand to each environment
        b_max_reward = self.b_reservations - self.s_reservations.min(-1)[0].unsqueeze_(-1) - self.epsilon
        s_max_reward = self.b_reservations.max(-1)[0].unsqueeze_(-1) - self.s_reservations - self.epsilon

        rewards_sellers, rewards_buyers = env.step()[1]

        loss_sellers = torch.abs(rewards_sellers - s_max_reward)
        loss_buyers = torch.abs(rewards_buyers - b_max_reward)

        # Mask the loss for agents who are already finished
        # TODO: Fix issue with masking values failing when used during backpropagation
        loss_sellers[self.done_sellers] = 0
        loss_buyers[self.done_buyers] = 0

        total_loss = torch.cat((loss_sellers, loss_buyers), -1)

        return total_loss
