import torch


class DummyAgent:
    def __init__(self, agent_id, number):
        self.number = number
        self.id = agent_id

    def get_action(self, observation: torch.Tensor):
        # time.sleep(0.0005)
        return self.number * observation.mean()  # dummy calculation


class DummyConstantAgent:
    def __init__(self, agent_id, number):
        self.number = number
        self.id = agent_id

    def get_action(self, observation: torch.Tensor):
        return self.number


class TimeAgent:
    def __init__(self, agent_id, role, reservation_price) -> None:
        self.agent_id = agent_id
        self.role = role
        self.reservation_price = reservation_price

    def get_action(self, observation):
        sign = 1.0 if self.role == "seller" else -1.0
        return (
            self.reservation_price
            + sign * (1 - observation[:, :, -1].mean()) * self.reservation_price
        )


class LinearAgent:
    def __init__(self, n_features) -> None:
        self.model = torch.nn.Linear(in_features=n_features, out_features=1)

    def get_action(self, observation):
        return self.model(observation).squeeze(1)
