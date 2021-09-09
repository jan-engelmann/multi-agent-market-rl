Documentation of the code
*************************

A quick overview
################
The Multi-Agent-Market has the following components:


**The environment:**

The environment module connects all other modules to creat a working engine. It starts by initialising all needed
settings. These consist of all the *agents*, the *market*, the *information setting*, the *exploration setting* and the
*reward setting*. In addition further initialisation settings can be defined in an optional keyword argument dictionary.
A detailed explanation of all the environment arguments will be provided in

After initialising all agents and settings, the environment takes care of computing a single time step in the market
environment. This consists of getting all current observations, computing all agent actions, computing all the deals
which got realized at the current time step *t* as well as the associated rewards.

*Side note: The realised deals will depend on the market setting and the achieved rewards will depend on the
reward setting*.

At the end of each time step *t*, the environment will return the current observations, the
current actions, the current rewards and the current agent status (done or active) of all agents at the time step
*t* as well as the next observations of all agents at time step *t+1*. In addition the environment returns a flag
indicating if the game has finished.

**The market:**

The market engine is in charge of computing all realized deals at the current time step *t*.
Currently only the **MarketMatchHiLo** class is implemented. This market engine calculates deals by matching the
highest buying offer with the lowest selling offer. The actual deal price is then taken as the mean value between the
matches buying offer and selling offer.

**The agents:**

The **AgentSetting** class is an abstract base class for all agents. It takes care of initialising the role and
reservation price as well as the action space of each agent. Again, custom agents can be created by adding classes
overwriting specific methods of the **AgentSetting** class.

Currently the following agents are already implemented:
    * DQNAgent
        This agent makes use of an artificial neural network in order to learn an optimal Q-function.
        This is achieved by making use of experience replay. The Q-Network is then iteratively updated
        using randomly selected experience minibatches.
    * HumanReplayAgent
        This agent replays data gathered from human experiments. All data is obtained from the
        "Trading-in-a-Black-Box" repository.
        https://github.com/ikicab/Trading-in-a-Black-Box/tree/f9d05b1a83882d41610638b0ceecfbb51cb05a85
    * ConstAgent
        This agent will always perform the same action.

**The info setting**:

The information setting dictates how much information agents get before deciding on an action. Currently all agents
always have access to the same amount of information.

Currently the following information settings are implemented:
    * BlackBoxSetting
        Every agent is aware of only its own last offer
    * OfferInformationSetting
        Every agent is aware of the best *N* offers of either side (buyer and seller) of the last
        round.
    * DealInformationSetting
        Every agent is aware of *N* deals of the last round
    * TimeInformationWrapper
        Wrapper to include the current in game time in the observation.

**The exploration setting:**

The exploration setting determines the evolution of the probability that an agent will perform a random action (perform
exploratory actions). All agents will make use of the same exploration setting.

Currently the following exploration setting is implemented:
    * LinearExplorationDecline
        Exploration probability declines linearly from an initial staring value down to a final
        value over the cors of *n* steps.

**The reward setting:**

Calculates the reward achieved by a given agent after closing a deal

Currently the following reward setting is implemented:
    * NoDealPenaltyReward
        Reward achieved by sellers is given by the difference of the deal and the reservation
        price of the seller. For buyers the reward is given buy the difference of the reservation
        and the deal price. In addition, buyers who spent more then N in game time steps without
        making a deal will receive a linearly increasing penalty (negative reward).

**The trainer:**
Finally we need to be able to train agents living in a given environment. To achieve this, we can build a trainer, that
plays through multiple episodes in order to train the agents.

Currently the following trainer is implemented:
    * DeepQTrainer
        Trainer following the philosophy of the DQN agents.

The Agent Module
################

The *AgentSetting* class provides the basic building blocks required by all agents. This includes the following::
    * Basic initialisation of the agent
        Every agent needs a role (buyer or seller), a reservation price, an observation space and an action space.
        The observation space is automatically determined according to the chosen information setting. The action space
        is automatically determined according to the agent role as well as the reservation prices of all other agents.
        Lastly every agent can optionally be assigned to a specific device (cpu or gpu).
    * All necessary methods (in the context of the DeepQTrainer)
        The methods *get_action*, *random_action*, *get_q_value*, *get_target*, *reset_target_network*,
        *save_model_weights* and *load_model_weights* all need to be callable in the context of the DeepQTrainer.
        However, their functionality is highly agent dependent. Therefore all methods implemented in the *AgentSetting*
        base class will raise *NotImplementedError*. Therefore every agent class has to overwrite the *get_action*
        method by means of inheritance of the *AgentSetting* class and thereby assigning the wanted behaviour of each
        method.

Ready to use agents
^^^^^^^^^^^^^^^^^^^

The **DQNAgent** class represents an agent who aims to approximate the optimal Q-function via a neural network. This is
done by having two identically initialised neural networks (the Q-network and the target-network). When learning, the
Q-network is updated accordingly to the 'ground truths' provided by the target-network. Periodically the target-network
is then reinitialised with the new weights from the Q-network.

The **HumanReplayAgent** class represents an agent capable of replaying human data gathered in a study. See this paper
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3131004 and/or the git directory 'HumanReplayData' containing the
the data as well as additional information such as player ids and reservation prices...

The **ConstAgent** class represents an agent who bids/asks a constant price during the entire game.

Creating your custom agent
^^^^^^^^^^^^^^^^^^^^^^^^^^

It is very easy to add your custom agent to the **agents.py** file. This is done in the following way

.. code-block:: Python
   :linenos:

   class MyCustomAgent(AgentSetting):
      def __init__(self,
                   role,
                   reservation,
                   in_features,
                   action_boundary,
                   device=torch.device('cpu'),
                   **kwargs):

         super(MyCustomAgent, self).__init__(role,
                                             reservation,
                                             in_features,
                                             action_boundary,
                                             device=device)

         <your custom code>

      def get_action(self, observation, epsilon=0.05):
         """
         Parameters
         ----------
         observation: torch.tensor
            Current observations
         epsilon: float, obtional (default=0.05)
            Probability for a random action

         Returns
         -------
         action: torch.tensor
         """


         <your custom code>

         return action

      ...

Every agent class that inherits from the **AgentSetting** class will have the following methods

* get_action(self, observation, epsilon=0.05)
   observation: torch.tensor, epsilon: float
* random_action(self, observation=None, epsilon=None)
   observation: torch.tensor, epsilon: float
* get_q_value(self, observation, actions=None)
   observation: torch.tensor, action: torch.tensor
* get_target(self, observation, agent_state=None)
   observation: torch.tensor, action: torch.tensor
* reset_target_network(self)
   No arguments
* save_model_weights(self)
   No arguments

By default all methods will raise *NotImplementedError*. In order for your custom agent to make proper use of these
methods, you will have to overwrite them. This is achieved by defining the exact same function in your custom agent
class as was done for the **get_action** method in the code example above. See the raw methods documentation for
detailed description of input and output types.

**Side note:** Not all agent types will necessarily make use of all methods. For instance zero intelligence agents will
never have to save weights. In order to circumvent the need to distinguish between agent types during training, we
suggest nonetheless implementing the 'useless' method as a dummy function consisting of a *pass* statement.

Furthermore all intelligent agents can make use of your custom neural network model via the **NetworkSetting** class.
First, you can define your own neural network model in the **network_models.py** file by inheriting the
**NetworkSetting** base class and overwriting the **define_network()** method. Be sure to return a pytorch neural
network model.

.. code-block:: Python
   :linenos:

   class MyCustomNetwork(NetworkSetting):
      def __init__(self, in_features, out_features, device=torch.device('cpu'), **kwargs):
         super(MyCustomNetwork, self).__init__(in_features,
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
      network = <your custom torch neural network>

      return network

In order to load your custom neural network into your intelligent agent, be sure to add a keyword argument
``"network_type"="MyCustomNetwork"`` to your agent kwargs dictionary, where "MyCustomNetwork" is a placeholder for the class
name of your actual network. Now you can load the neural network as follows:

.. code-block:: Python
   :linenos:

   import network_models as network_models

   class MyCustomAgent(AgentSetting):
      def __init__(self, role, reservation, in_features, ..., device, **kwargs):
         ...
         network_type = kwargs.pop("network_type", <name of default network>)
         out_features = len(self.action_space)
         network_builder = getattr(network_models, network_type)(
            in_features, out_features, device=self.device, **kwargs
         )

         my_custom_network = network_builder.get_network()

The resulting network will already be located on the correct device (cpu or gpu). Also, if kwargs contains a
"load_weights_path" keyword, the model weights will be loaded from the provided path.

Now you can make use of your custom agent in the same way as you use the other agents.

The Market Module
#################

The **BaseMarketEngine** class forms the basis of all market engines. It takes care of initialising the number of
sellers and the number of buyers as well as the max time duration of one episode/game and implements the buyer, seller
and deal histories.

Ready to use market engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **MarketMatchHiLo** class is a ready implemented matching algorithm matching the highest bidding buyer with the
lowest asking seller, second highest bidder with the second lowest asking seller, and so on. The actual deal price is
then computed as the mean between the matches bidding and asking prices.

Creating your custom engine
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Again you can creat your own market engine with custom deal matching by inheriting the **BaseMarketEngine** class and
overwriting the **calculate_deals** method. However you are confined to returning the calculated deals for sellers and
buyers separately as torch.tensors.

.. code-block:: Python
   :linenos:

   class MyCustomMarketEngine(BaseMarketEngine):

      def __init__(self, n_sellers, n_buyers, device=torch.device('cpu'), **kwargs):
         super(MyCustomMarketEngine, self).__init__(n_sellers,
                                                    n_buyers,
                                                    device=device,
                                                    **kwargs)

         <your custom code>

      def calculate_deals(self, s_actions, b_actions):
         """

         Parameters
         ----------
         s_actions: torch.Tensor
            Has shape n_sellers
         b_actions: torch.Tensor
            Has shape n_buyers

         Returns
         -------
         deals_sellers: torch.Tensor
            Has shape n_sellers
         deals_buyers: torch.Tensor
            Has shape n_buyers
         """

         <your custom code>

         return deals_sellers, deals_buyers

Again, once you have implemented you custom market engine in the *markets.py* file, you can make use of the new market
engine as usual in the environment.

The Information Setting Module
##############################

The **InformationSetting** class forms the basis of all information settings. It provides access to all environment
variables as well as the **get_states(...)** method which can be overwritten in order to define your custom information
states.

Ready to use information settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **BlackBoxSetting** class provides a setting where the information state of each agent consists of only its last
action.

The **OfferInformationSetting** class provides a setting where the information state of each agent consists of the best
N offers on both sides (buyer and seller). Therefore each agent will have the same 2*N offers as its information state.

The **DealInformationSetting** class provides a setting where the information state of each agent consists of the best
N deals of the last round. Again, every agent will have the same information state.

The **TimeInformationWrapper** class takes any other information setting class and adds the current time step to the
information state.

Creating your custom information setting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As stated above, you can creat your custom information setting by inheriting the **InformationSetting** base class and
overwriting the **get_states(...)** method. Again, the return object is the limiting factor to your creativity. It is
expected, that the **get_states(...)** returns a *torch.tensor* object of size (n_agents, n_features). Where the
n_agents dimension first contains all sellers and then contains all buyers. n_features represents the number of
distinct information feature each agent gets.

.. code-block:: Python
   :linenos:

   class MyCustomInformationClass(InformationSetting):

          def __init__(self, env, **kwargs):
              super(MyCustomInformationClass, self).__init__(env)

              <your custom code>

          def get_states(self):
              """
              Returns
              -------
              total_info: torch.tensor
                  Return total_info as tensor with shape (n_agents, n_features) where
                  n_features == number of infos Observations are ordered in the same way
                  as res in MultiAgentEnvironment.get_actions().
                  total_info[:n_sellers, :] contains all observations for the seller agents
                  total_info[n_sellers:, :] contains all observations for the buyer agents
              """

              <your custom code>

              return total_info

Again, once you have implemented your custom information setting in the **info_setting.py**, you can make use of it in
the same way as the ready to use info settings.

The Exploration Setting Module
##############################

The **ExplorationSetting** class forms the base class of all exploration settings. Currently it does absolutely nothing
besides initialising the *epsilon* class variable representing the probability for a random (exploratory) action.

Ready to use exploration settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **LinearExplorationDecline** class provides a setting, where the exploration vale declines linearly from a given
starting value to a given final value over a given number of steps.

Creating your custom exploration setting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As always, you can creat your custom exploration setting in the **exploration_setting.py** file by inheriting from the
**ExplorationSetting** base class and overwriting the **update()** method. Again, the 'updated' probability to perform
a random action must be assigned to the *epsilon* class variable in order to have an effect.

.. code-block:: Python
   :linenos:

   class MyCustomExploration(ExplorationSetting):

      def __init__(self, **kwargs):
         super(MyCustomExploration, self).__init__(**kwargs)

         <your custom code>

      def update(self):

         <your custom code>

         self.epsilon = <your new epsilon value>

Now you can make use of your newly made exploration setting in the same way as the ready made ones can be implemented
into the environment.

The Reward Setting Module
#########################

The **RewardSetting** class forms the basis of all reward settings. It provides access to all environment variables.

Ready to use reward settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **NoDealPenaltyReward** class provides a reward setting, where buyers receive a linearly growing penalty if they do
not manage to close a deal after N rounds. This has the objective of enticing the agents to act quickly.

Creating your custom reward setting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to creat your custom reward setting in the **reward_setting.py** file, you must inherit the **RewardSetting**
base class and overwrite the **seller_reward()** and the **buyer_reward** methods. Again, you are restricted in the
shape of the return object. Both methods need to return a torch.tensor object with shape (n_sellers,) and (n_buyers,)
respectively.

.. code-block:: Python
   :linenos:

   class MyCustomReward(RewardSetting):

      def __init__(self, env, **kwargs):
         super(MyCustomReward, self).__init__(env)

         <your custom code>

      def seller_reward(self, seller_deals):
         """
         Parameters
         ----------
         seller_deals: torch.tensor
            Has shape (n_sellers,)

         Returns
         -------
         seller_rewards: torch.tensor
            Has shape (n_sellers,)
         """

         <your custom code>

         return seller_rewards

      def buyer_reward(self, buyer_deals):
         """
         Parameters
         ----------
         buyer_deals: torch.tensor
            Has shape (n_buyers,)

         Returns
         -------
         buyer_rewards: torch.tensor
            Has shape (n_buyers,)
         """

         <your custom code>

         return buyer_rewards

Now you can make use of your custom reward setting in the same manner as the ready implemented reward settings are used.

The Neural Network Model Module
###############################

The **NetworkSetting** class is intended to provide easy to use modularity with regards to the neural networks used by
intelligent agents. Following the same architecture as the previous modules, the **NetworkSetting** class forms the base
class, providing all needed functionalities.

Ready to use network models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **SimpleExampleNetwork** class provides a simple neural network for the purpose of being an example.

Creating your custom neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to define your own neural network in the **network_models.py** file, which can be used by intelligent agents,
you have to inherit the **NetworkSetting** base class and overwrite the **define_network** methode. The class variables
**in_features** and **out_features** provide the input and output shape of your neural network. The method must return
a torch neural network (torch.nn).

.. code-block:: Python
   :linenos:

   class MyCustomNetworkModel(NetworkSetting):

      def __init__(self, in_features, out_features, device=torch.device('cpu'), **kwargs):
         super(SimpleExampleNetwork, self).__init__(in_features,
                                                    out_features,
                                                    device=device,
                                                    **kwargs)

      def define_network(self):
         """
         Returns
         -------
         network: torch.nn
            The wanted neural network
         """

         <your custom code>

         return network

Now you can import your newly created neural network model into your custom made agent class by making use of the
**get_network** methode. Add something in the lines of this to your init function:

.. code-block:: Python
   :linenos:

   network_builder = getattr(network_models, "MyCustomNetworkModel")(
       in_features, out_features, device=self.device, **kwargs
   )

   my_network = network_builder.get_network()


Raw module functions
####################

.. automodule:: agents
    :members:
    :inherited-members:
    :undoc-members:
    :special-members: __init__

.. automodule:: network_models
    :members:
    :inherited-members:
    :undoc-members:
    :special-members: __init__

.. automodule:: info_setting
    :members:
    :inherited-members:
    :undoc-members:
    :special-members: __init__

.. automodule:: exploration_setting
    :members:
    :inherited-members:
    :undoc-members:
    :special-members: __init__

.. automodule:: reward_setting
    :members:
    :inherited-members:
    :undoc-members:
    :special-members: __init__

.. automodule:: environment
    :members:
    :inherited-members:
    :undoc-members:
    :special-members: __init__

.. automodule:: markets
    :members:
    :inherited-members:
    :undoc-members:
    :special-members: __init__

.. automodule:: trainer
    :members:
    :inherited-members:
    :undoc-members:
    :special-members: __init__
