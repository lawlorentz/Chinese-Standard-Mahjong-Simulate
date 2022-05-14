from itertools import count
from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_right

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from data_augment import data_augment

import os


def safe_del(path):
    if os.path.exists(path):
        os.remove(path)
    else:
        print("The file does not exist")


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class MahjongGBDataset(Dataset):

    def __init__(self, begin=0, end=1, augment=False):
        import json
        with open('data/count.json') as f:
            self.match_samples = json.load(f)
        if augment == True:
            self.match_samples = [i*12 for i in self.match_samples]
            self.augmented_matchs = []

        self.total_matches = len(self.match_samples)
        self.total_samples = sum(self.match_samples)
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        self.match_samples = self.match_samples[self.begin: self.end]
        self.matches = len(self.match_samples)
        self.samples = sum(self.match_samples)
        self.augment = augment
        self.cache_to_match = np.zeros([2048])
        self.match_to_cache = np.zeros([self.matches])
        for i in range(self.matches):
            self.match_to_cache[i] = -1
        self.count = 0
        self.is_fulled = False
        self.cache = {'obs': [], 'mask': [], 'act': []}
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a

        pass

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        sample_id = index - self.match_samples[match_id]
        if self.match_to_cache[match_id] == -1:
            if self.augment == False:
                d = np.load('data/%d.npz' % (match_id))
            else:
                # 只保存2048个%d_augmented.npz，要添加就得先删除一个
                if not os.path.exists('data/%d_augmented.npz' % (match_id)):
                    if len(self.augmented_matchs) >= 2048:
                        need_del = self.augmented_matchs[0]
                        self.augmented_matchs = self.augmented_matchs.pop(0)
                        safe_del('data/%d_augmented.npz' % (need_del))
                    data_augment(match_id)
                    self.augmented_matchs.append(match_id)
                d = np.load('data/%d_augmented.npz' % (match_id))
            if self.is_fulled == False:
                self.cache['obs'].append(d['obs'])
                self.cache['mask'].append(d['mask'])
                self.cache['act'].append(d['act'])
            else:
                self.cache['obs'][self.count] = d['obs']
                self.cache['mask'][self.count] = d['mask']
                self.cache['act'][self.count] = d['act']
                pre_match = int(self.cache_to_match[self.count])
                self.match_to_cache[pre_match] = -1
            self.match_to_cache[match_id] = self.count
            self.cache_to_match[self.count] = match_id
            self.count = self.count+1
            if self.count == 2048:
                self.count = 0
                self.is_fulled = True
        tmp = int(self.match_to_cache[match_id])
        return self.cache['obs'][tmp][sample_id], self.cache['mask'][tmp][sample_id], self.cache['act'][tmp][sample_id]


# mydataset = MahjongGBDataset(0.3,0.8)
# x=mydataset[400]
# pass
