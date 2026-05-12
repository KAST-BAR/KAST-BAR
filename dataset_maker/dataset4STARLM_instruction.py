from einops import rearrange
import pickle
from pathlib import Path
import random
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader, Sampler

standard_1020 = [
    'AF9', 'AF7', 'F9', 'FP1', 'AF5', 'AF3', 'F7', 'F5', 'F3', \
    'FPZ', 'AF1', 'AFZ', 'AF2', 'F1', 'FZ', 'F2', \
    'FP2', 'AF4', 'AF6', 'AF8', 'AF10', 'F10', 'F4', 'F6', 'F8', \
    'FT9', 'FT7', 'A1', 'T9', 'T7', 'TP9', 'TP7', \
    'FC5', 'FC3', 'C5', 'C3', 'CP5', 'CP3', \
    'FC1', 'FCZ', 'FC2', 'C1', 'CZ', 'C2','CP1', 'CPZ', 'CP2', \
    'FC4', 'FC6', 'C4', 'C6','CP4', 'CP6', \
    'FT8', 'FT10', 'T8', 'T10', 'A2', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'PO9', 'PO7', 'PO5', 'PO3', \
    'P1', 'PZ', 'P2', 'PO1', 'POZ', 'PO2', \
    'P4', 'P6', 'P8', 'P10', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O9', 'O1', 'I1', 'OZ', 'IZ', 'O2', 'O10', 'I2', 'pad'
]


class EEGTextLoader(Dataset):
    def __init__(self, files, sampling_rate=200, GPT_training=False):
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.GPT_training = GPT_training
        self.seed_label_map = {
            'H': 0,
            'N': 1,
            'S': 2,
        }

    def __len__(self):
        return len(self.files)

    def std_norm(self, x):
        mean = torch.mean(x, dim=(0, 1), keepdim=True)
        std = torch.std(x, dim=(0, 1), keepdim=True)
        x = (x - mean) / std
        return x

    def get_chans(self, ch_names):
        chans = []
        for ch_name in ch_names:
            chans.append(standard_1020.index(ch_name))
        return chans

    def _extract_label(self, sample):
        if "y" in sample:
            y = sample["y"]
            if isinstance(y, str):
                return self.seed_label_map.get(y, None)
            elif isinstance(y, (int, float, np.integer, np.floating)):
                return int(y)
            elif isinstance(y, (list, tuple, np.ndarray)):
                return int(y[0]) if len(y) > 0 else None
        if "label" in sample:
            label = sample["label"]
            if isinstance(label, (list, tuple, np.ndarray)):
                if len(label) > 0:
                    return int(label[0] - 1)
            elif isinstance(label, (int, float, np.integer, np.floating)):
                return int(label)
        return None

    def __getitem__(self, index):
        sample = pickle.load(open(self.files[index], "rb"))
        data = sample["X"]
        dataset_name = sample["dataset_name"]
        sample_name = sample["sample_name"]
        label = self._extract_label(sample)
        if isinstance(dataset_name, str) and 'TUEV' in dataset_name and label is not None:
            if label >= 1:
                label = label - 1
        ch_names = sample["ch_names"]
        data = torch.FloatTensor(data)
        data = self.std_norm(data)
        time = data.size(1) // 200
        chanel_num = data.size(0)
        blocksize = chanel_num * time
        input_time = [i % 64 for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        X = torch.zeros((blocksize, 200))
        X[:data.size(0)] = data

        if not self.GPT_training:
            Y_freq = torch.zeros((blocksize, 100))
            Y_raw = torch.zeros((blocksize, 200))
            x_fft = torch.fft.fft(data, dim=-1)
            amplitude = torch.abs(x_fft)
            amplitude = self.std_norm(amplitude)
            Y_freq[:data.size(0)] = amplitude[:, :100]
            Y_raw[:data.size(0)] = self.std_norm(data)

        input_chans = list(ch_names) * time
        input_chans.extend(['pad'] * (blocksize - data.size(0)))
        input_chans = torch.IntTensor(self.get_chans(input_chans))
        input_time.extend([0] * (blocksize - data.size(0)))
        input_time = torch.IntTensor(input_time)
        input_mask = torch.ones(blocksize)
        input_mask[data.size(0):] = 0
        if label is None:
            label = -1
        return X, label, input_chans, input_time, input_mask.bool(), dataset_name, sample_name


class GroupByChannelHashBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=12, drop_last=True, seed=42, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = seed
        self.shuffle = shuffle
        files = getattr(dataset, 'files', None)
        if files is None:
            raise ValueError(
                'Dataset must have a files attribute to use GroupByChannelHashBatchSampler')
        groups = {}
        for idx, file_path in enumerate(files):
            path_obj = Path(file_path)
            path_parts = path_obj.parts
            try:
                train_regroup_idx = path_parts.index('train_regroup')
                if train_regroup_idx + 2 < len(path_parts):
                    subdir = path_parts[train_regroup_idx + 1]
                    channel_hash_dir = path_parts[train_regroup_idx + 2]
                    group_key = f"{subdir}/{channel_hash_dir}"
                    groups.setdefault(group_key, []).append(idx)
                else:
                    group_key = str(path_obj.parent)
                    groups.setdefault(group_key, []).append(idx)
            except ValueError:
                if len(path_obj.parts) >= 2:
                    parent_parent = path_obj.parent.parent.name if path_obj.parent.parent.name else ""
                    parent = path_obj.parent.name if path_obj.parent.name else ""
                    group_key = f"{parent_parent}/{parent}" if parent_parent else parent
                    groups.setdefault(group_key, []).append(idx)
                else:
                    group_key = str(path_obj.parent)
                    groups.setdefault(group_key, []).append(idx)
        self.groups = groups
        self.epoch = 0
        self._rebuild_batches()

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._rebuild_batches()

    def _rebuild_batches(self):
        rnd = random.Random(self.base_seed + self.epoch)
        batches = []
        for group_key, indices in self.groups.items():
            if self.shuffle:
                rnd.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch_indices)
        if self.shuffle:
            rnd.shuffle(batches)
        self.batches = batches
        print(
            f"Created {len(batches)} batches from {len(self.groups)} channel hash groups (shuffle={self.shuffle})")


class DistributedGroupByChannelHashBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=12, drop_last=True, seed=42, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = seed
        self.epoch = 0
        self.shuffle = shuffle
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0
        self.num_replicas = num_replicas
        self.rank = rank
        files = getattr(dataset, 'files', None)
        if files is None:
            raise ValueError(
                'Dataset must have a files attribute to use DistributedGroupByChannelHashBatchSampler')
        groups = {}
        for idx, file_path in enumerate(files):
            path_obj = Path(file_path)
            path_parts = path_obj.parts
            try:
                train_regroup_idx = path_parts.index('train_regroup')
                if train_regroup_idx + 2 < len(path_parts):
                    subdir = path_parts[train_regroup_idx + 1]
                    channel_hash_dir = path_parts[train_regroup_idx + 2]
                    group_key = f"{subdir}/{channel_hash_dir}"
                    groups.setdefault(group_key, []).append(idx)
                else:
                    group_key = str(path_obj.parent)
                    groups.setdefault(group_key, []).append(idx)
            except ValueError:
                if len(path_obj.parts) >= 2:
                    parent_parent = path_obj.parent.parent.name if path_obj.parent.parent.name else ""
                    parent = path_obj.parent.name if path_obj.parent.name else ""
                    group_key = f"{parent_parent}/{parent}" if parent_parent else parent
                    groups.setdefault(group_key, []).append(idx)
                else:
                    group_key = str(path_obj.parent)
                    groups.setdefault(group_key, []).append(idx)
        self.groups = groups
        print(
            f"DistributedGroupByChannelHashBatchSampler: Found {len(groups)} unique channel hash groups, rank={rank}/{num_replicas}")
        self._rebuild_batches()

    def __iter__(self):
        for batch in self.my_batches:
            yield batch

    def __len__(self):
        return len(self.my_batches)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._rebuild_batches()

    def _rebuild_batches(self):
        rnd = random.Random(self.base_seed + self.epoch)
        all_batches = []
        for group_key, indices in self.groups.items():
            if self.shuffle:
                rnd.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch_indices)
        if self.shuffle:
            rnd.shuffle(all_batches)
        total = len(all_batches) - (len(all_batches) % self.num_replicas)
        all_batches = all_batches[:total]
        self.my_batches = []
        for i, batch in enumerate(all_batches):
            if i % self.num_replicas == self.rank:
                self.my_batches.append(batch)
        print(
            f"[Rank {self.rank}] Created {len(self.my_batches)} batches (total: {len(all_batches)} batches)")


def statistics_label_distribution(files, seed=42):
    temp_loader = EEGTextLoader(files)
    dataset_label_stats = defaultdict(lambda: defaultdict(int))
    file_label_map = {}
    print("Computing label distribution...")
    for idx, file_path in enumerate(files):
        try:
            sample = pickle.load(open(file_path, "rb"))
            dataset_name = sample.get("dataset_name", "unknown")
            label = temp_loader._extract_label(sample)
            if isinstance(dataset_name, str) and 'TUEV' in dataset_name and label is not None:
                if label >= 1:
                    label = label - 1
            if label is None:
                label = -1
            dataset_label_stats[dataset_name][label] += 1
            file_label_map[file_path] = (dataset_name, label)
        except Exception as e:
            print(f"  Warning: failed to read {file_path}: {e}")
            continue
    result = {}
    print("\n" + "=" * 80)
    print("Label distribution:")
    print("=" * 80)
    for dataset_name, label_counts in sorted(dataset_label_stats.items()):
        result[dataset_name] = dict(label_counts)
        total_samples = sum(label_counts.values())
        print(f"\nDataset: {dataset_name}")
        print(f"  Total samples: {total_samples}")
        print(f"  Per-label counts:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = count / total_samples * 100
            print(f"    Label {label}: {count} samples ({percentage:.2f}%)")
    print("=" * 80 + "\n")
    return result, file_label_map


def downsample_by_label(files, file_label_map, downsample_strategy='median',
                        downsample_threshold=None, seed=42, verbose=True):
    dataset_label_files = defaultdict(lambda: defaultdict(list))
    for file_path in files:
        if file_path in file_label_map:
            dataset_name, label = file_label_map[file_path]
            dataset_label_files[dataset_name][label].append(file_path)
    rnd = random.Random(seed)
    downsampled_files = []
    if verbose:
        print("\n" + "=" * 80)
        print("Downsampling...")
        print("=" * 80)
    for dataset_name, label_files_dict in sorted(dataset_label_files.items()):
        label_counts = [len(files_list) for files_list in label_files_dict.values()]
        if downsample_strategy == 'median':
            threshold = int(np.median(label_counts)) if label_counts else 0
        elif downsample_strategy == 'mean':
            threshold = int(np.mean(label_counts)) if label_counts else 0
        elif downsample_strategy == 'min':
            threshold = min(label_counts) if label_counts else 0
        elif downsample_strategy == 'fixed':
            threshold = downsample_threshold if downsample_threshold is not None else max(label_counts)
        else:
            raise ValueError(f"Unknown downsample strategy: {downsample_strategy}")
        if verbose:
            print(f"\nDataset: {dataset_name}")
            print(f"  Threshold: {threshold} (strategy: {downsample_strategy})")
        for label, files_list in sorted(label_files_dict.items()):
            original_count = len(files_list)
            if original_count > threshold:
                rnd.shuffle(files_list)
                selected_files = files_list[:threshold]
                downsampled_files.extend(selected_files)
                if verbose:
                    print(f"  Label {label}: {original_count} -> {threshold} (removed {original_count - threshold})")
            else:
                downsampled_files.extend(files_list)
                if verbose:
                    print(f"  Label {label}: {original_count} (unchanged)")
    if verbose:
        print(f"\nDownsample done: {len(files)} -> {len(downsampled_files)} files")
        print("=" * 80 + "\n")
    return downsampled_files


def create_data(batch_size=12, dataset_dir=None, ddp=False, ddp_rank=0, ddp_world_size=1, group_by_hash=True, num_workers=2, split='train',
                enable_label_statistics=True, enable_downsample=False, downsample_strategy='median', downsample_threshold=None, downsample_seed=42):
    print(f'Preparing dataloader for split={split}...')
    shuffle = split == 'train'
    dataset_dir = Path(dataset_dir)
    all_files = []
    split_dir_direct = dataset_dir / split
    if split_dir_direct.exists() and split_dir_direct.is_dir():
        print(f"  Single-dataset layout under: {dataset_dir.name}")
        pkl_files = list(split_dir_direct.rglob('*.pkl'))
        all_files.extend(pkl_files)
        print(f"  Found {split}: {len(pkl_files)} files")
    else:
        found_any = False
        for dataset_subdir in dataset_dir.iterdir():
            if dataset_subdir.is_dir():
                split_dir = dataset_subdir / split
                if split_dir.exists() and split_dir.is_dir():
                    pkl_files = list(split_dir.rglob('*.pkl'))
                    all_files.extend(pkl_files)
                    print(f"  Found {dataset_subdir.name}/{split}: {len(pkl_files)} files")
                    found_any = True
                else:
                    print(f"  Warning: missing {dataset_subdir.name}/{split}, skipped")
        if not found_any:
            print(f"  Warning: no {split} split found under dataset subdirectories")
    print(f"Total {split} files: {len(all_files)}")
    if len(all_files) == 0:
        raise ValueError(
            f"No pickle files for split={split}. Expected either:\n"
            f"  1) Multi-dataset: dataset_dir/<NAME>/{split}/...\n"
            f"  2) Single dataset: dataset_dir/{split}/...")
    file_label_map = None
    if split == 'train' and (enable_label_statistics or enable_downsample):
        label_stats, file_label_map = statistics_label_distribution(all_files, seed=downsample_seed)
    if split == 'train' and enable_downsample:
        if file_label_map is None:
            label_stats, file_label_map = statistics_label_distribution(all_files, seed=downsample_seed)
        all_files = downsample_by_label(
            all_files,
            file_label_map,
            downsample_strategy=downsample_strategy,
            downsample_threshold=downsample_threshold,
            seed=downsample_seed,
            verbose=True
        )
        print(f"After downsampling: {len(all_files)} files")
    dataset_train = EEGTextLoader(all_files)
    print('Done.')
    if ddp:
        if group_by_hash:
            batch_sampler = DistributedGroupByChannelHashBatchSampler(
                dataset_train,
                batch_size=batch_size,
                drop_last=True,
                seed=42,
                num_replicas=ddp_world_size,
                rank=ddp_rank,
                shuffle=shuffle
            )
        else:
            batch_sampler = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=shuffle
            )
    else:
        if group_by_hash:
            batch_sampler = GroupByChannelHashBatchSampler(
                dataset_train,
                batch_size=batch_size,
                drop_last=True,
                seed=42,
                shuffle=shuffle
            )
        else:
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset_train)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset_train)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size=batch_size, drop_last=True
            )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    return data_loader_train
