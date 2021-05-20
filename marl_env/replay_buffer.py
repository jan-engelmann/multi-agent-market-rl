import numpy as np
from tianshou.data import Batch

from tianshou.data import ReplayBuffer as TianshouRBuffer
from tianshou.data.batch import _create_value, _alloc_by_keys_diff

from typing import Any, List, Tuple, Union, Optional


class ReplayBuffer(TianshouRBuffer):
    def __init__(
        self,
        size: int,
        stack_num: int = 1,
        ignore_obs_next: bool = False,
        save_only_last_obs: bool = False,
        sample_avail: bool = False,
        **kwargs: Any,  # otherwise PrioritizedVectorReplayBuffer will cause TypeError
    ) -> None:
        super(ReplayBuffer, self).__init__(
            size,
            stack_num=stack_num,
            ignore_obs_next=ignore_obs_next,
            save_only_last_obs=save_only_last_obs,
            sample_avail=sample_avail,
            **kwargs,
        )

    # Overwrite the add attribute to allow for rewards of type torch.Tensor
    def add(
        self, batch: Batch, buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Add a batch of data into replay buffer.
        :param Batch batch: the input data batch. Its keys must belong to the 7
            reserved keys, and "obs", "act", "rew", "done" is required.
        :param buffer_ids: to make consistent with other buffer's add function; if it
            is not None, we assume the input batch's first dimension is always 1.
        Return (current_index, episode_reward, episode_length, episode_start_index). If
        the episode is not finished, the return value of episode_length and
        episode_reward is 0.
        """
        # preprocess batch
        b = Batch()
        for key in set(self._reserved_keys).intersection(batch.keys()):
            b.__dict__[key] = batch[key]
        batch = b
        assert {"obs", "act", "rew", "done"}.issubset(batch.keys())
        stacked_batch = buffer_ids is not None
        if stacked_batch:
            assert len(batch) == 1
        if self._save_only_last_obs:
            batch.obs = batch.obs[:, -1] if stacked_batch else batch.obs[-1]
        if not self._save_obs_next:
            batch.pop("obs_next", None)
        elif self._save_only_last_obs:
            batch.obs_next = (
                batch.obs_next[:, -1] if stacked_batch else batch.obs_next[-1]
            )
        # get ptr
        if stacked_batch:
            rew, done = batch.rew[0], batch.done[0]
        else:
            rew, done = batch.rew, batch.done
        ptr, ep_rew, ep_len, ep_idx = list(map(lambda x: x, self._add_index(rew, done)))
        try:
            self._meta[ptr] = batch
        except ValueError:
            stack = not stacked_batch
            # batch.rew = batch.rew.astype(float) --> not compatible with torch.Tensor
            batch.done = batch.done.astype(bool)
            if self._meta.is_empty():
                self._meta = _create_value(batch, self.maxsize, stack)  # type: ignore
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize, stack)
            self._meta[ptr] = batch
        return ptr, ep_rew, ep_len, ep_idx
