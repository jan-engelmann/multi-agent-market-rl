Quick guide to using the environment
***********************************

.. _agent_dict:

The agent dictionary
####################

The agent dictionary is used to inform the environment of how many agents and of what type and with what specifications
will be participating. The agent dictionary has a nested dictionary structure. More specifically, every agent with the
role *seller* and every agent with the role *buyer* are collected in separate dictionaries.

``agent_dict = {'sellers': seller_agent_dict, 'buyers': buyer_agent_dict}``

The **seller_agent_dict** is again a nested dictionary made up of *n* (number of selling agent configurations) single
agent dictionaries (same holds for the **buyer_agent_dict**).

``seller_agent_dict = {1: single_agent_seller_1, ..., n: single_agent_seller_n}
buyer_agent_dict = {1: single_agent_buyer_1, ..., m: single_agent_buyer_m}``

Finally a single agent dictionary is made up of the following ``key:value`` pairs:

**Mandatory key:value pairs**
    * 'type' (str)
        The name of the wanted agent class object
    * 'reservation' (int)
        The reservation price for this agent
**Optional key:value pairs**
    * 'multiplicity' (int)
        The multiplicity count of the specific agent implementation (default=1)
    * **kwargs
        Additional keyword arguments specific to the chosen agent type

``single_agent_dict = {'type': 'MyAgentType', 'reservation': 12, 'multiplicity': 3, **kwargs}``

The agent types currently implemented have the following type specific ``kwargs``:

**DQNAgent**
    * network_type: str, optional (default="SimpleExampleNetwork")
        Name of network class implemented in network_models.py
    * q_lr: float (default=0.001)
        Learning rate provided to the Q-Network optimizer
    * save_weights_directory: str (default="../saved_agent_weights/default_path/{self.agent_name}/")
        Directory to where model weights will be saved to
    * save_weights_file: str (default="default_test_file.pt")
        File name of the saved weights. Must be a .pt or .pth file
    * load_weights_path: str (default=False)
        If a path is provided, agent will try to load pretrained weights from there

**ConstAgent**
    * const_price: int (default=Mean value of action space)
        The constant asking / bidding price

**HumanReplayAgent**
    * data_type: str (default='new_data')
        Data set used (new_data or old_data). See the git directory 'HumanReplayData'
    * treatment: str (default='FullLimS')
        Market treatment used. See https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3131004
    * id: int (default=954)
        Player id. Must match with agent 'role', 'reservation', 'data_type' and 'treatment'. See the .csv files in
        the git directory 'HumanReplayData/data_type'

Example of a complete agent dictionary:

.. code-block:: Python
   :linenos:

   agent_dict = {'sellers': {1: {'type': 'DQNAgent', 'reservation': 5}},
                 'buyers': {1: {'type': 'ConstAgent',
                                'reservation': 15,
                                'const_price': 7},
                            2: {'type': 'ConstAgent',
                                'reservation': 20,
                                'const_price': 18}}}

This would result in one **DQNAgent** seller with a reservation price of 5 and two **ConstAgent** buyers with
reservation price and bidding price of (15, 7) and (20, 18) respectively. If all goes well, the **DQNAgent** should
learn to sell his product to the **ConstAgent** bidding 18.

.. _env:

Initialising the environment
############################

Now we can start by initialising the environment. For this we need the following arguments:

* Agent dictionary (dict)
    See :ref:`agent_dict`
* Market setting (str)
    You can specify what market engine to use by passing the market class name as a string. Optionally you can specify
    the market engine by providing a pre-initialised market class object. Currently the only implemented market engine
    is **MarketMatchHiLo**
* Information setting (str)
    You can specify what information setting to use by passing the information setting class name as a string.
    Optionally you can specify the information setting by providing a pre-initialised information setting class object.
    Currently the implemented information settings are **BlackBoxSetting**, **OfferInformationSetting**,
    **DealInformationSetting** and **TimeInformationWrapper**
* Exploration setting (str)
    You can specify what exploration setting to use by passing the exploration setting class name as a string.
    Optionally you can specify the exploration setting by providing a pre-initialised exploration setting class object.
    Currently the only implemented exploration setting is **LinearExplorationDecline**
* Reward setting (str)
    You can specify what reward setting to use by passing the reward setting class name as a string. Optionally you can
    specify the reward setting by passing a pre-initialised reward setting class object. Currently the only implemented
    reward setting is **NoDealPenaltyReward**
* Optional kwargs
    We can fine tune all market, information, exploration and reward settings by providing a keyword argument dictionary
    for every individual setting. In addition we can specify on what device the environment and on what device the
    agent networks should operate. The currently implemented keyword arguments are the following:

    * market_settings
        * Global
            * max_steps: int (default=30)
                Maximum number of time steps before the game is reset.
        * MarketMatchHiLo
            None
    * info_settings
        * Global
            None
        * BlackBoxSetting
            None
        * OfferInformationSetting
            * n_offers: int (default=1)
                Number of offers to see. For instance, 5 would mean the agents see the best 5 bids and asks
        * DealInformationSetting
            * n_deals: int (default=1)
                Number of deals to see
        * TimeInformationWrapper
            * base_setting: InformationSetting object (default="BlackBoxSetting")
                The base information setting to add time to
    * exploration_settings
        * Global
            None
        * LinearExplorationDecline
            * initial_expo: float (default=1.0)
                Initial exploration probability
            * n_expo_steps: int (default=100000)
                Number of time steps over which the exploration rate will decrease linearly
            * final_expo: float (default=0.0)
                Final exploration rate
    * reward_settings
        * Global
            None
        * NoDealPenaltyReward
            * no_deal_max: int (default=10)
                Number of allowed time steps without making a deal before being punished
    * device: list (default=['cpu', 'cpu'])
        Responsible for providing GPU support. The environment is thought to run on two GPUs. One GPU for the
        environment and one for the agent optimization. If provided should be a list of two GPU devices. First device
        will be for the environment, second device will be for agent networks. **'cpu'** refers to the current CPU
        device. **'cuda'** refers to the current GPU device. In order to differentiate between different GPU devices
        use **'cuda:i'** where **i** is the respective GPU index (starting from zero)

Example of a complete keyword argument dictionary fine tuning the environment settings as well as initialising an
environment compatible with the chosen keyword arguments:

.. code-block:: Python
   :linenos:

   settings_kwargs = {'market_settings': {'max_steps': 45},
                      'info_settings': {'n_offers': 10},
                      'exploration_settings': {'initial_expo': 0.95,
                                               'n_expo_steps': 1e6,
                                               'final_expo': 1e-5},
                      'reward_settings': {'no_deal_max': 5},
                      'device': ['cuda:0', 'cuda:1']}

   env = MultiAgentEnvironment(agent_dict,
                               'MarketMatchHiLo',
                               'OfferInformationSetting',
                               'LinearExplorationDecline',
                               'NoDealPenaltyReward',
                               **settings_kwargs)

Main functionalities of the environment
########################################

There are two main functionalities of the environment.

* env.reset()
    Will reset the environment to its initial settings.
* env.step(random_action=False)
    Will perform one single time step forward in the environment. If *random_action=True* all agents will perform a
    random action. In addition this function returns the current observations, current actions, current rewards,
    next observations, agent states (active or finished) and a 'done flag' indicating if the current game has finished
    or not.

Quick guide to using the DeepQTrainer
*************************************

The **DeepQTrainer** is a ready built trainer copying the training procedure of the *Human-level control through
deep reinforcement learning* paper, see https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf .
Therefore the application of this trainer is mainly useful in the context of training **DQNAgents**.

Initialising the DeepQTrainer
#############################

The main idea of this trainer, is to use a replay buffer from which samples are randomly picked in order to train the
agents. The **DeepQTrainer** uses the following arguments:

* env: environment object
    The current environment class object in which the agents are living. See :ref:`env`
* memory_size: int
    Total size of the ReplayBuffer. This corresponds to the total number of actions memorised for every agent.
* replay_start_size: int
    Number of ReplayBuffer slots to be initialised with a uniform random policy before learning starts

In addition, there are some optional keyword arguments to allow for further fine tuning of the trainer

* discount: float, optional (default=0.99)
    Multiplicative discount factor for Q-learning update
* update_frq: int, optional (default=100)
    Frequency (measured in episode/game counts) with which the target network is updated
* max_loss_history: int, optional (default=None)
    Number of previous episodes for which the loss will be saved for monitoring
    None --> All episode losses are saved
* max_reward_history: int, optional (default=None)
    Number of previous episodes for which the rewards will be saved for monitoring
    None --> All episode rewards are saved
* max_action_history: int, optional (default=None)
    Number of previous episodes for which the actions will be saved for monitoring
    None --> All episode actions are saved
* loss_min: int, optional (default=-5)
    Lower-bound for the loss to be clamped to
* loss_max: int, optional (default=5)
    Upper-bound for the loss to be clamped to
* save_weights: bool, optional (default=False)
    If true, all agent weights will be saved to the respective directory specified by the agent in question

**Example initialisation of the DeepQTrainer**

.. code-block:: Python
   :linenos:

   env = MultiAgentEnvironment(...)
   mem_size = 10000
   start_size = 500

   trainer = DeepQTrainer(env, mem_size, start_size)

Training the agents
###################

It is very easy to start the training process. We just need to make use of the **train(...)** methode. This takes the
following arguments:

* n_episodes: int
    Number of episodes/games to train for
* batch_size: int
    Batch size used to update the agents network weights (Number of action/result pairs used in one learning step)

The trainer will return some statistics. Namely the average loss history for every agent, the average reward history
for every agent and the action history of every agent. All three are returned as a list of torch.tensors.

**Example use of the train(...) method**

.. code-block:: Python
   :linenos:

    n_episodes = 100000
    batch_size = 32
    total_loss, total_rew, actions = trainer.train(n_episodes, batch_size)

Remark on implementing a custom training loop
#############################################

One can implement a multitude of training loops tailored to ones custom made reinforcement learning agents.
However, a replay buffer is often needed. Therefore we slightly modified the **tianshou.data.ReplayBuffer** to allow
for all data to be of type *torch.tensor* and also modified the random sampling to return batches containing the
following keywords:

* obs (torch.tensor)
   All current observations
* act (torch.tensor)
   All current actions
* rew (torch.tensor)
   All current rewards
* done (bool)
   Bool indicating if the episode/game has ended or not
* obs_next (torch.tensor)
   All observations from the next round (t+1)
* a_states (torch.tensor)
   All current agent states (active or done)

The original **tianshou.data.ReplayBuffer** can be found here https://github.com/thu-ml/tianshou

The modified ReplayBuffer can be directly imported from the **marl_env** directory and supports these main features

.. code-block:: Python
   :linenos:

   from tianshou.data import Batch
   from replay_buffer import ReplayBuffer

   # Initialise the buffer with a fixed size
   buffer = ReplayBuffer(size=memory_size)

   # Add a history batch to the ReplayBuffer --> we make use of tianshou.data.Batch
   history_batch = Batch(
                obs=obs,
                act=act,
                rew=rew,
                done=done,
                obs_next=obs_next,
                a_states=a_states,
            )
   buffer.add(history_batch)

   # Sample a random minibatch of transitions from the replay buffer
   batch_data, indices = buffer.sample(batch_size=batch_size)

Further Usage Examples
######################

For further usage examples pleas visit the git **Examples** directory.
