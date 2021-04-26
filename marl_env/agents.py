import torch


class DummyAgent:
    def __init__(self, agent_id, number):
        self.number = number
        self.id = agent_id

    def get_action(self, observation: torch.Tensor):
        # time.sleep(0.0005)
        return self.number * observation.mean()  # dummy calculation