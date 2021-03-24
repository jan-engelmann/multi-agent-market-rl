# multi-agent-market-rl
Creating a multi agent reinforcement learning environment for two sided auction markets.

## Current Environment
Market Env from course: https://github.com/asikist-ethz/market_rl 

## Project description
https://docs.google.com/document/d/1ItXiqNpgwbX3JohVJuKzJxrjqOm5dG_Qn--kN4AlXZ8/edit

## Empirical data analysis
Data [Barbara]

- Download [link](https://www.dropbox.com/s/3j4f9cbzh3imfr7/data.csv?dl=0)
- Data description [link](https://mfr.osf.io/render?url=https://osf.io/8a97e/?direct%26mode=render%26action=download%26mode=render)
- https://github.com/ikicab/Trading-in-a-Black-Box


## Strong previous RL projects
https://github.com/Alehud/RL_for_markets
https://github.com/ekarais/RLFM

## Thomas code sketch

```python
class AgentEnv():

    def init(global):
        self.global(check are all expected keys in)

    def step(action):
        self.global_env(id, action)
        while(not self.global.are_keys_in()): #join asynchronous taks
            obs, rew, done = self.global_env.request_observation(id)
            break
        return obs, rew, done

    def reset():
        while (not self.global.has_everyone_reset()):
            obs =  self.global.get_reset_observation()
            return obs

```

## Thomas compute advice
I suggest you run it in Euler (if it is only CPU) or Leonhard cluster if it is either CPU or GPU.
For GPU to see any serious speedup you would need a good mini batch procedure for learning.
This means that each agent will be learning and generating many RL experiences in parallel.
Unless you plan to use many machines, this will be efficient only if your environment is vectorised in GPU, i.e. the environment step computations happens in GPU and calculate batches of observations/rewards for batches of action inputs in parallel.
Also some RL models are more easy to parallelize, i.e. Asynchronous Actor Critic A3C.

See [cluster_usage.pdf](https://github.com/jan-engelmann/multi-agent-market-rl/blob/main/misc/cluster_usage.pdf)
