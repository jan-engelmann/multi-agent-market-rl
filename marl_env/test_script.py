import torch
import time

from matplotlib import pyplot as plt
from marl_env.trainer import DeepQTrainer
from marl_env.environment import MultiAgentEnvironment


if __name__ == "__main__":
    agent_dict = {
        'sellers': {
            1: {
                'type': 'HumanReplayAgent',
                'reservation': 68,
                'multiplicity': 1,
                **{'id': 1022},
            },
        },
        'buyers': {
            1: {
                'type': 'DQNAgent',
                'reservation': 128,
                'multiplicity': 1,
                **{'lr': 0.005,}
            },
        }
    }

    kwargs = {
        'market_settings': {},
        'info_settings': {},
        'exploration_settings': {'n_expo_steps': 100,
                                 'final_expo': 0.05},
        'reward_settings': {'no_deal_max': 1}
    }

    env = MultiAgentEnvironment(
        agent_dict,
        'MarketMatchHiLo',
        'BlackBoxSetting',
        'LinearExplorationDecline',
        'NoDealPenaltyReward',
        **kwargs
    )

    n_episodes = 200
    batch_size = 16
    mem_size = 150
    start_size = 100

    train_params = {'discount': 0.3,
                    'update_frq': 250,
                    'loss_min': -50,
                    'loss_max': 50,
                    'save_weights': True}

    trainer = DeepQTrainer(env, mem_size, start_size, **train_params)

    tt = time.time()
    total_loss, total_rew, actions = trainer.train(n_episodes, batch_size)
    print("Total time in s: ", time.time() - tt)

    tot_loss = torch.stack(total_loss, dim=0).transpose(0, 1).detach()
    tot_rew = torch.stack(total_rew, dim=0).squeeze().transpose(0, 1).detach()
    actions = torch.stack(actions, dim=0).transpose(0, 1).detach()

    plt.figure()
    plt.title("Average Loss")
    plt.semilogy(tot_loss[1, :], label="DQA Buyer")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Average Reward")
    plt.plot(tot_rew[0, :], 'o', label="Const Seller")
    plt.plot(tot_rew[1, :], 'o', label="DQA Buyer")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("Last actions")
    plt.plot(actions[0, :], 'o', label="Const Seller")
    plt.plot(actions[1, :], 'o', label="DQA Buyer")
    plt.show()
