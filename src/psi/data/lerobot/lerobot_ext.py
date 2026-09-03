from __future__ import annotations
from typing import Any, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    from psi.config.data_lerobot import LerobotDataConfig
    # from psi.config.data_simple import SimpleDataConfig

import random
import torch
from psi.data.lerobot.compat import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from psi.utils import resolve_path
from psi.config.transform import LerobotRepackTransform


def _episode_subset(root_dir, repo_ids, split, val_ratio, seed):
    subsets = {}
    for repo_id in repo_ids:
        meta = LeRobotDatasetMetadata(repo_id, resolve_path(f"{root_dir}/{repo_id}"))
        episodes = sorted(meta.episodes)
        random.Random(f"{seed}:{repo_id}").shuffle(episodes)
        num_val = min(len(episodes) - 1, max(1, int(len(episodes) * val_ratio + 0.5)))
        subsets[repo_id] = sorted(episodes[:num_val] if split == "val" else episodes[num_val:])
    return subsets


def _fix_v21_subset_indices(dataset):
    if dataset.episodes is None:
        return
    starts = torch.zeros(dataset.meta.total_episodes, dtype=torch.long)
    ends = torch.zeros_like(starts)
    offset = 0
    for episode in dataset.episodes:
        starts[episode] = offset
        offset += dataset.meta.episodes[episode]["length"]
        ends[episode] = offset
    dataset.episode_data_index = {"from": starts, "to": ends}


class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """ A wrapper around LeRobotDataset to support multiple datasets.
    """

    def __init__(
        self, 
        data_cfg: LerobotDataConfig, 
        split: str = "train"
    ):
        repo_ids = data_cfg.train_repo_ids if split == "train" else data_cfg.val_repo_ids
        first_repo = repo_ids[0] if isinstance(repo_ids, list) else repo_ids
        episode_map = (
            _episode_subset(data_cfg.root_dir, repo_ids, split,
                            data_cfg.episode_val_ratio, data_cfg.episode_split_seed)
            if data_cfg.episode_val_ratio is not None
            else None
        )
        dataset_meta = LeRobotDatasetMetadata(first_repo, resolve_path(f"{data_cfg.root_dir}/{first_repo}"))
        assert isinstance(data_cfg.transform.repack, LerobotRepackTransform)
        delta_timestamps = data_cfg.transform.repack.delta_timestamps(dataset_meta.fps)

        if len(repo_ids) > 1:
            root_dir = data_cfg.root_dir
            lerobot_dataset_class = MultiLeRobotDataset
            episodes = episode_map
        else:
            repo_ids = first_repo
            root_dir = resolve_path(f"{data_cfg.root_dir}/{first_repo}")
            lerobot_dataset_class = LeRobotDataset
            episodes = episode_map[first_repo] if episode_map is not None else None

        self.base_dataset = lerobot_dataset_class(
            repo_ids,# type: ignore
            root=root_dir,
            episodes=episodes,
            delta_timestamps=delta_timestamps, # type: ignore
            image_transforms=None,
        )
        for dataset in getattr(self.base_dataset, "_datasets", [self.base_dataset]):
            _fix_v21_subset_indices(dataset)
        self._cache = {}

    def __getitem__(self, idx) -> dict:
        return self.base_dataset[idx]
    
    def __len__(self):
        return len(self.base_dataset)

    @property
    def episode_data_index(self):
        return self.base_dataset.episode_data_index # type: ignore

    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes
    
    @property
    def num_frames(self):
        return self.base_dataset.num_frames
    
    @property
    def meta(self):
        return self.base_dataset.meta # type: ignore

    @property
    def stats(self):
        return self.base_dataset.stats if type(self.base_dataset) == MultiLeRobotDataset else self.base_dataset.meta.stats # type: ignore
