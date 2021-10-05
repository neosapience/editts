import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import re

def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def intersperse_emphases(emphases):
    for n in range(len(emphases)):
        emphases[n][0] = 2 * emphases[n][0]
        emphases[n][1] = 2 * emphases[n][1] + 1
    return emphases


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model