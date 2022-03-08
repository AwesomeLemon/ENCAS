'''
Separate a part of the train for setting batch norms. If you have the full ImageNet train set on your machine, this
is not necessary.
'''

import numpy as np
import os
from os.path import join

import random

import shutil

from pathlib import Path

random.seed(3179)
np.random.seed(3179)

n_select = 10

in_path = join('', 'train')
out_path1 = join('', 'train_part')

try:
    Path(out_path1).mkdir()
except FileExistsError:
    shutil.rmtree(out_path1)
    Path(out_path1).mkdir()

for path_class in os.scandir(in_path):
    if not path_class.is_dir():
        continue
    name_cur = path_class.name
    out_dir1_cur = join(out_path1, name_cur)
    for p in [out_dir1_cur]:
        Path(p).mkdir(exist_ok=True)

    file_names = np.array([f for f in os.scandir(path_class) if not f.is_dir()])
    permutation = np.arange(len(file_names))
    np.random.shuffle(permutation)
    idx1 = permutation[:n_select]
    print(name_cur, idx1)
    for f in file_names[idx1]:
        shutil.copy(f.path, join(out_dir1_cur, f.name))