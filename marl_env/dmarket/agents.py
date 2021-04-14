import numpy as np

class MarketAgent:
    """
    Market agent implementation to be used in market environments.

    Attributes
    ----------
    role: str, 'buyer' or 'seller'
    reservation_price: float
        Must be strictly positive.

    name: str, optional (default=None)
        Name of the market agent. If not given, a random one will be generated.
        Note: this will usually not be the agent id used in the market engine.
    """
    def __init__(self, role, reservation_price, name=None):
        if not role in ['buyer', 'seller']:
            raise ValueError("Role must be either buyer or seller")
        if reservation_price <= 0:
            raise ValueError("Reservation price must be positive")

        self.role = role
        self.reservation_price = reservation_price
        if not name:
            randstring = "%04x" % np.random.randint(16**4)
            cls = type(self).__name__[0:4]
            letter = role[0].upper()
            name = f"{cls}_{letter}{reservation_price}_{randstring}"

        self.name = name

    def get_offer(self, observation):
        """
        Returns offer given an observations.

        Parameters
        ----------
        observation: array_like
            An element of some observation space defined by the used
            information setting.

        Returns
        -------
        offer: float
            Offer to made to the market.
        """
        raise NotImplementedError


class ConstantAgent(MarketAgent):
    """Agent that always offers its reservation price."""
    def get_offer(self, observation):
        return self.reservation_price


class FactorAgent(MarketAgent):
    """
    Abstract agent class that determines an offer range based on a
    multiplicative factor.

    Children of this class will take an argument ``max_factor`` and use this
    together with its reservation price to determine an interval to offer in.
    For a seller, the agent will have the range ``[r, (1+max_factor)*r]`` with
    ``r`` the reservation price. For a buyer, this interval would be
    ``[(1-max_factor)*r, r]``.

    Parameters
    ----------
    max_factor: float, optional (default=0.5)
        Must be between 0 and 1.

    Attributes
    ----------
    _s: int
        A sign derived from the role of the agent, +1 means seller, -1 means
        buyer.
    _c: float
        A factor used to compute the interval range.
    _a: float
        Lower bound of the offer range.
    _b: float
        Upper bound of the offer range.
    """

    def __init__(self, role, reservation_price, name=None, max_factor=0.5):
        self.max_factor = max_factor
        super().__init__(role, reservation_price, name)

        r = reservation_price
        self._s = (-1 if role == 'buyer' else 1)
        self._c = (1 + self._s * max_factor)
        self._a = min(r, self._c*r) # minimum agent can offer
        self._b = max(r, self._c*r) # maximum agent can offer


class UniformRandomAgent(FactorAgent):
    """
    Random agent that offers uniformly random prices.

    This agent will take an argument ``max_factor`` and use this together
    with its reservation price to determine the interval to use for sampling
    offers. For a seller, the agent will make uniform random offers in the
    range ``[r, (1+max_factor)*r]`` with ``r`` the reservation price. For a
    buyer, this interval would be ``[(1-max_factor)*r, r]``.

    Parameters
    ----------
    max_factor: float, optional (default=0.5)
        Must be between 0 and 1.
    """

    def get_offer(self, observation):
        return np.random.uniform(self._a, self._b)


class TimeDependentAgent(FactorAgent):
    """
    Abstract helper class to create agents that have time-dependent strategies.
    """
    def get_offer(self, observation):
        if not isinstance(observation, tuple):
            raise ValueError("Expected tuple observation!")
        obs  = observation[0]
        time = observation[1]
        return self.compute_offer(obs, time)

    def compute_offer(self, observation, time):
        """
        Compute the offer based on observation and time.
        """
        raise NotImplementedError


class TimeLinearAgent(TimeDependentAgent):
    """
    Agent that linearly decreases/increases its offer price.

    This agent starts with a high offer and linearly decreases/increases its
    price until it reaches its reservation price.

    Parameters
    ----------
    max_factor: float, optional (default=0.5)
        Must be between 0 and 1, determines the offer range of the agent.
    noise: float, optional (default=1.0)
        The standard deviation of the noise added to the computed price.
    max_steps: int, optional (default=20)
        Number of steps until the agent offers its reservation price. This
        determines how quickly the agent lowers/increases his price.
    """

    def __init__(self, role, reservation_price, name=None, max_factor=0.5,
                 noise=1.0, max_steps=20):
        super().__init__(role, reservation_price, name, max_factor)
        self.max_steps = max_steps
        self.noise = noise
        self._slope = -self._s * (self._b - self._a)/self.max_steps

    def compute_offer(self, observation, time):
        t = min(time,  self.max_steps)
        noise = np.random.normal(scale=self.noise)
        return self._c*self.reservation_price + t*self._slope + noise


class GymRLAgent(FactorAgent):
    """
    A market agent with reinforcement learning model.

    This class serves as a wrapper for gym RL models and serves two purposes:
    1. Standardize action space for RL models;
    2. Make trained RL models applicable under different market situations.

    The second point is achieved through normalization of input observations.
    This makes it possible for an agent that was trained as a seller to
    operate as a buyer. It also enables agents to function properly across
    markets with different price scales.

    Parameters
    ----------
    model: object, optional (default=None)
        Trained baselines model to use for predictions. It needs to have the
        method ``predict(observation) -> (action, state)``.

    discretization: int, optional (default=20)
        The number of different offers the agent can make. This determines the
        action space of the agent.

    max_factor: int
        A factor of the reservation price that determines the range of prices
        the agent can offer. See ``UniformRandomAgent``.
    """
    def __init__(self, role, reservation_price, name=None, model=None,
                 discretization=20, max_factor=0.5):
        self.model = model
        self.discretization = discretization
        self._N = discretization
        super().__init__(role, reservation_price, name, max_factor)

    def get_offer(self, observation):
        if not self.model:
            raise RuntimeError("Current agent does not have a model")
        action = self.model.predict(self.normalize(observation))[0]
        return self.action_to_price(action)

    def normalize(self, observation):
        """
        Normalize the prices in observations according to reservation price.

        This function will serve to scale all observations based on the agent's
        reservation price and role. An observation should contain nonnegative
        prices. A small value corresponds to prices close to the agent's
        reservation price, while large values correspond to attractive offers.

        To preserve symmetry between sellers and buyers, this function is
        discontinuous in 0, since the information settings represent no
        offers/no information with zero.

        Parameters
        ----------
        observation: array_like
            An element of the observation space determined by the information
            setting.

        Returns
        -------
        normalized_observation: array_like
            Scaled observation based on agent's reservation price and role.
        """
        return np.heaviside(observation, 0) * self._s * \
                (observation - self.reservation_price)/self.reservation_price

    def action_to_price(self, action):
        """
        Convert an action in the action space of the agent to a market price.

        This function uniformly discretizes the price range of the agent. An
        action close to zero should yield a conservative offer, i.e., close
        to the reservation price, while a large value for action gives more
        aggressive offers.

        Parameters
        ----------
        action: int
            The action is an integer ``0 <= action < discretization``.

        Returns
        -------
        price: float
            The price corresponding to the action.
        """
        l = action - self._N/2
        m = self._N/2
        return ((m - l*self._s)*self._a + (m + l*self._s)*self._b)/self._N
