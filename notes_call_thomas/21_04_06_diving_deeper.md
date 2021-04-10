# Call with more details
## Output
- Fast RL environment with parallel training
- Use custom torch model
- train with 10-50 agents
- put environment logic in linear algebra to use torch fast computations
- use dummy dimensions to get math right

## Thomas will send
- linalg physics paper with lucas and nino
- ipynb with code examples

## Techniques
- use hirarchical model
- try RNN

## Regarding continuous and discrete spaces
- people think in discrete spaces
- some think tahts easier to train
- he prefers continous spaces
- FOR NOW: start with discrete spaces

## Tech
- See ipynb created during call
- checkout tianshou

## Problem with current environemnt
- slow training
- RNN agents didnt work
- bad scaling behaviour
- only works with very few agents
- sequential processing of agent agents (implement async)
- only got dqn to work and therefore only on discrete spaces


## Notes from Ben
Environments: DDPG // rllib --> Multiagents

Goal: Scale up environment such that we can train 10-50 agents.

Use hirarchical actions: Split Participation yes/no --> if yes, what price. This reduces the action space.
General, reduce the result space by adding as many constraints as possible directly into the NN.

Batch-Size in RL is the number of parallel environment runs. 

Most importent part:
Write computation of deal-price (Or what is needed to compute the reward) as matrix equation.
We want to have all environments in one matrix dim: (n_env, n_agents, n_features, ...) ---> reward will be dim: (n_env, n_agents, 1) ???

Computational ideas:
- Use interaction matrices
- Use expansion / compression of dimensions. 
- Laplace Operator can be interessting...
- ---> Implement in pytorch