Hi everyone,

Thanks for sharing the GitHub notes.

Please find the update from my side.

Notebook

I have updated the notebook file (see attached file) and organised a bit the content.
The proposed environment is vectorised, multi-agent and can be multi-sampled.
You may see how I quickly parallelise across agents and samples inside the step function by only relying on torch.

Paper
The example of code vectorisation can be found below:
- Element-wise/Individual equations are reported in the following paper in equation (23) p.9:
https://arxiv.org/pdf/2006.09773.pdf

- The corresponding vectorised code of the matrix equations can be found in:
https://github.com/asikist/nnc/blob/6324597e1264eb9a0e72ba9c0706f9421f7dc874/nodec_experiments/sirx/sirx.py#L207

Other notes:
 reward will be dim: (n_env, n_agents, 1) ???

Exactly. We don’t aggregate rewards across samples/agents.
We use rewards to calculate a learning loss per agent and then we aggregate learning losses over mini-batches/many samples to speed up training for each agent.

For RL, I agree we start from tianshou, as I am also more familiar with it and perhaps can support you more.
Also, the other libraries  I had in mind seem to focus on single agent examples mostly, so they may require a lot of changes.

Hope the above help a bit, and please let me know if you have any other notes/questions!

Thanks for your time,
Best
Thomas