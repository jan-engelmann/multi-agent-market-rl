import torch
import torch.multiprocessing as mp
import time
import itertools
import pandas as pd
from tqdm import tqdm


class Agent:
    def __init__(self, number):
        self.number = number

    def get_action(self, observation: torch.tensor):
        # time.sleep(0.0005)
        return self.number * observation.mean()  # dummy calculation


def get_agent_actions(agent, observation):
    return agent.get_action(observation)


class MultiAgentEnvironment:
    def __init__(self, n_sellers, n_buyers, n_environments, n_features, worker_pool):

        self.n_sellers = n_sellers
        self.n_buyers = n_buyers
        self.n_agents = n_sellers + n_buyers
        self.max_n_deals = min(n_buyers, n_sellers)
        self.max_group_size = max(n_buyers, n_sellers)
        self.n_environments = n_environments  # batch_size
        self.n_features = n_features

        # TODO implement proper agent
        self.sellers = [Agent(num) for num in torch.rand(n_sellers, n_environments)]
        self.buyers = [Agent(num) for num in torch.rand(n_buyers, n_environments)]

        self.worker_pool = worker_pool
        self.reset()

    def reset(self):
        self.s_reservations = torch.rand(self.n_sellers)
        self.b_reservations = torch.rand(self.n_buyers)
        self.past_actions = []
        self.observations = [
            torch.rand(self.n_agents, self.n_environments, self.n_features)
        ]  # filling with initial observation

    def get_actions(self):
        # TODO improve distributed calculation
        agent_observation_tuples = [
            (agent, obs)
            for agent, obs in zip(
                itertools.chain(self.sellers, self.buyers), self.observations[-1]
            )
        ]
        res = torch.stack(
            self.worker_pool.starmap(
                get_agent_actions,
                agent_observation_tuples,
            )
        ).T
        return torch.split(res, [self.n_sellers, self.n_buyers], dim=1)

    def calculate_deals(self, s_actions, b_actions):
        # sort actions of sellers and buyers
        # using mechanism that highest buying offer is matched with lowest selling offer
        s_actions_sorted, s_actions_indices = s_actions.sort()
        b_actions_sorted, b_actions_indices = b_actions.sort(descending=True)

        # get mask for all deals that happen
        bid_offer_diffs = torch.zeros(self.n_environments, self.max_group_size)
        bid_offer_diffs[:, : self.max_n_deals] = (
            b_actions_sorted[:, : self.max_n_deals]
            - s_actions_sorted[:, : self.max_n_deals]
        )
        no_deal_mask = (
            bid_offer_diffs <= 0
        )  # if true => no deal, if the bid is lower than the offer no deal happens

        s_realized_sorted_deals = s_actions_sorted.clone()
        s_realized_sorted_deals[no_deal_mask[:, : self.n_sellers]] = 0

        b_realized_sorted_deals = b_actions_sorted.clone()
        b_realized_sorted_deals[no_deal_mask[:, : self.n_buyers]] = 0

        # calculating deal prices
        sorted_deal_prices = torch.zeros(self.n_environments, self.max_group_size)
        sorted_deal_prices[:, : self.max_n_deals] = (
            b_realized_sorted_deals[:, : self.max_n_deals]
            + s_realized_sorted_deals[:, : self.max_n_deals]
        ) / 2.0

        # getting the realized prices in the original ordering for buyers&sellers, no deal means 0
        deals_buyers_orginal = sorted_deal_prices[:, : self.n_buyers].gather(
            1, b_actions_indices.argsort(1)
        )

        deals_sellers_orginal = sorted_deal_prices[:, : self.n_sellers].gather(
            1, s_actions_indices.argsort(1)
        )
        return deals_sellers_orginal, deals_buyers_orginal

    def calculate_rewards(self, deals_sellers, deals_buyers):
        rewards_sellers = deals_sellers - self.s_reservations[None, :]
        rewards_buyers = self.b_reservations[None, :] - deals_buyers
        return rewards_sellers, rewards_buyers

    def store_observations(self, s_actions, b_actions, deals_sellers, deals_buyers):
        # TODO actually implement valid observations
        self.observations.append(
            torch.rand(self.n_agents, self.n_environments, self.n_features)
        )

    def step(self):
        s_actions, b_actions = self.get_actions()
        deals_sellers, deals_buyers = self.calculate_deals(s_actions, b_actions)
        rewards_sellers, rewards_buyers = self.calculate_rewards(
            deals_sellers, deals_buyers
        )
        self.store_observations(s_actions, b_actions, deals_sellers, deals_buyers)

        # print(deals_buyers, deals_sellers, rewards_buyers, rewards_sellers, sep="\n")


if __name__ == "__main__":
    n_sellers = 50
    n_buyers = 50
    n_environments = 15
    n_processes = [1, 2, 4, 6, 8]
    times = []
    n_iterations = 5
    n_features = 100
    n_steps = 50

    for n_proc in tqdm(n_processes, desc="n_processes"):
        for it in tqdm(range(n_iterations), desc="n_iterations", leave=False):
            duration = 0.0
            start = time.time()
            with mp.Pool(n_proc) as pool:
                env = MultiAgentEnvironment(
                    n_sellers, n_buyers, n_environments, n_features, pool
                )
                for i in range(n_steps):
                    env.step()
            end = time.time()
            duration += end - start
        times.append(duration / n_iterations)

    df = pd.DataFrame(dict(n_processes=n_processes, time=times))
    print(df)