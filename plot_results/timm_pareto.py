'''
find pareto front of timm models, save it 10 times to make my code think there are 10 seeds (this is needed for plotting)
'''
import json
import numpy as np
import os
import yaml

import utils
from utils import NAT_LOGS_PATH
from utils_pareto import is_pareto_efficient
from pathlib import Path

path_test_data = os.path.join(NAT_LOGS_PATH, 'timm_all/pretrained/0/output_distrs_test/info.json')
loaded = json.load(open(path_test_data))

flops = np.array(loaded['flops'])
acc = np.array(loaded['test'])
err = 100 - acc

all_objs_cur = np.vstack((err, flops)).T
pareto_best_cur_idx = is_pareto_efficient(all_objs_cur)
flops_pareto = list(reversed(flops[pareto_best_cur_idx].tolist()))
acc_pareto = list(reversed(acc[pareto_best_cur_idx].tolist()))

print(flops_pareto, acc_pareto)

out_dir = os.path.join(utils.NAT_PATH, 'logs_classification', 'timm_all')

for i in range(10):
    out_dir_cur = os.path.join(out_dir, str(i))
    Path(out_dir_cur).mkdir(exist_ok=True)
    out_file_cur = os.path.join(out_dir_cur, 'data.yml')
    yaml.safe_dump(dict(flops=flops_pareto, test=acc_pareto), open(out_file_cur, 'w'))