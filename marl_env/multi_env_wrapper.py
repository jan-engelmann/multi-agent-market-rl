import torch
from marl_env.environment import MultiAgentEnvironment


class MultiEnvWrapper:
    def __init__(self):
        self.device = None
        pass

    def _set_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.n_gpu = torch.cuda.device_count()
        else:
            self.device = 'cpu'

