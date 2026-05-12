from einops import rearrange
import pickle
from pathlib import Path
import random
import itertools

import torch
from torch.utils.data import Dataset, DistributedSampler, DataLoader, ConcatDataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch.utils.data import Sampler
from model.standard_1020_chorder import gen_stair_scale_mask, ms_ch_dict
from model.standard_1020_chorder import remove_unused_ch

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
    'O9', 'O1', 'I1', 'OZ', 'IZ', 'O2', 'O10', 'I2', 'pad'\
]  # add 'pad'


class PickleLoader(Dataset):
    def __init__(self, files, block_size=1024, sampling_rate=200, GPT_training=False):
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.GPT_training = GPT_training
        self.standard_1020_dict = {name: idx for idx, name in enumerate(standard_1020)}

    def __len__(self):
        return len(self.files)

    def std_norm(self, x):
        mean = torch.mean(x, dim=(0, 1), keepdim=True)
        std = torch.std(x, dim=(0, 1), keepdim=True)
        x = (x - mean) / std
        return x

    def get_chans(self, ch_names):
        return [self.standard_1020_dict[ch_name] for ch_name in ch_names]

    def __getitem__(self, index):
        sample = pickle.load(open(self.files[index], "rb"))
        data = sample["X"]
        ch_names = sample["ch_names"]
        data = torch.FloatTensor(data)
        time = data.size(1) // 200
        input_time = [i for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        X = torch.zeros((self.block_size, 200))
        X[:data.size(0)] = data

        if not self.GPT_training:
            Y_freq = torch.zeros((self.block_size, 100))
            Y_raw = torch.zeros((self.block_size, 200))
            x_fft = torch.fft.fft(data, dim=-1)
            amplitude = torch.abs(x_fft)
            amplitude = self.std_norm(amplitude)
            Y_freq[:data.size(0)] = amplitude[:, :100]
            Y_raw[:data.size(0)] = self.std_norm(data)

        input_chans = list(ch_names) * time
        input_chans.extend(['pad'] * (self.block_size - data.size(0)))
        input_chans = torch.IntTensor(self.get_chans(input_chans))
        input_time.extend([0] * (self.block_size - data.size(0)))
        input_time = torch.IntTensor(input_time)
        input_mask = torch.ones(self.block_size)
        input_mask[data.size(0):] = 0
        ms_bool, ms_ch = remove_unused_ch(ms_ch_dict, ch_names)
        ms_bool_bp = ms_bool.copy()
        ms_bool = list(itertools.chain(*ms_bool))
        num_ms_ch = ms_bool.count(True)

        if self.GPT_training:
            mask_type = 'stair_scale'
            if mask_type == 'stair':
                num_ms_ch = ms_bool_bp[-1].count(True)
                len_gpt_mask = num_ms_ch * time
                gpt_mask = torch.ones(len_gpt_mask, len_gpt_mask).view(1, len_gpt_mask, len_gpt_mask)
                for i in range(time):
                    if i == 0:
                        continue
                    gpt_mask[:, (i - 1) * num_ms_ch:i * num_ms_ch, i * num_ms_ch:] = 0
            elif mask_type == 'stair_scale':
                gpt_mask = gen_stair_scale_mask(ms_ch, time)
                gpt_mask = torch.from_numpy(gpt_mask)
            num_chans = len(ch_names)
            ms_bool = torch.tensor(ms_bool, dtype=torch.bool)
            return X, input_chans, input_time, input_mask.bool(), gpt_mask.bool(), num_chans, data.size(0), ms_bool

        return X, Y_freq, Y_raw, input_chans, input_time, input_mask.bool()


def collate_fn(batch):
    X_eeg, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens, ms_bool = zip(*batch)
    max_len = max(m.size(0) for m in gpt_mask)
    padded_masks = [
        F.pad(mask, (0, 0, 0, max_len - mask.size(0)), value=1)
        for mask in gpt_mask
    ]
    padded_masks = [
        F.pad(mask, (0, max_len - mask.size(1), 0, 0), value=0)
        for mask in padded_masks
    ]
    gpt_mask = torch.stack(padded_masks).unsqueeze(1)
    (X_eeg, input_chans, input_time, input_mask, ms_bool) = map(torch.stack, (X_eeg, input_chans, input_time, input_mask, ms_bool))
    (num_chans, num_tokens) = map(torch.tensor, (num_chans, num_tokens))
    return X_eeg, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens, ms_bool


class EEGTextLoader(Dataset):
    def __init__(self, files, block_size=1024, sampling_rate=200, GPT_training=False):
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.GPT_training = GPT_training

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

    def __getitem__(self, index):
        sample = pickle.load(open(self.files[index], "rb"))
        data = sample["X"]
        dataset_name = sample["dataset_name"]
        sample_name = sample["sample_name"]
        ch_names = sample["ch_names"]
        data = torch.FloatTensor(data / 100)
        time = data.size(1) // 200
        input_time = [i for i in range(time) for _ in range(data.size(0))]
        data = rearrange(data, 'N (A T) -> (A N) T', T=200)
        X = torch.zeros((self.block_size, 200))
        X[:data.size(0)] = data

        if not self.GPT_training:
            Y_freq = torch.zeros((self.block_size, 100))
            Y_raw = torch.zeros((self.block_size, 200))
            x_fft = torch.fft.fft(data, dim=-1)
            amplitude = torch.abs(x_fft)
            amplitude = self.std_norm(amplitude)
            Y_freq[:data.size(0)] = amplitude[:, :100]
            Y_raw[:data.size(0)] = self.std_norm(data)

        input_chans = list(ch_names) * time
        input_chans.extend(['pad'] * (self.block_size - data.size(0)))
        input_chans = torch.IntTensor(self.get_chans(input_chans))
        input_time.extend([0] * (self.block_size - data.size(0)))
        input_time = torch.IntTensor(input_time)
        input_mask = torch.ones(self.block_size)
        input_mask[data.size(0):] = 0
        return X, Y_freq, Y_raw, input_chans, input_time, input_mask.bool(), dataset_name, sample_name


class GroupByChannelHashBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=12, drop_last=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = seed
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
            rnd.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch_indices)
        rnd.shuffle(batches)
        self.batches = batches
        print(
            f"Created {len(batches)} batches from {len(self.groups)} channel hash groups")


class DistributedGroupByChannelHashBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=12, drop_last=True, seed=42, num_replicas=None, rank=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.base_seed = seed
        self.epoch = 0
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
            rnd.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch_indices)
        rnd.shuffle(all_batches)
        total = len(all_batches) - (len(all_batches) % self.num_replicas)
        all_batches = all_batches[:total]
        self.my_batches = []
        for i, batch in enumerate(all_batches):
            if i % self.num_replicas == self.rank:
                self.my_batches.append(batch)
        print(
            f"[Rank {self.rank}] Created {len(self.my_batches)} batches (total: {len(all_batches)} batches)")


def create_data(batch_size=12, dataset_dir=None, ddp=False, ddp_rank=0, ddp_world_size=1, group_by_hash=True, num_workers=2):
    print('Preparing dataloader...')
    print(f"Loading from directory: {dataset_dir}")
    files = Path(dataset_dir).rglob('*.pkl')
    files = [file for file in files]
    print(f"Found {len(files)} pickle files")
    dataset_train = EEGTextLoader(files)
    print('Done.')

    if ddp:
        if group_by_hash:
            batch_sampler = DistributedGroupByChannelHashBatchSampler(
                dataset_train,
                batch_size=batch_size,
                drop_last=True,
                seed=42,
                num_replicas=ddp_world_size,
                rank=ddp_rank
            )
        else:
            batch_sampler = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
    else:
        if group_by_hash:
            batch_sampler = GroupByChannelHashBatchSampler(
                dataset_train,
                batch_size=batch_size,
                drop_last=True,
                seed=42
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset_train)
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
