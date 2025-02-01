# The main file for prediction
import os, sys
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import itertools
import argparse
import json
import tqdm

import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist

sys.path.append(".")
from config.config import config, update_config
from utils import exp_utils
from eval_utils import eval_utils
from eval_utils.task_inference_predict import Task

def perform_query(annotations, config):
    for cuid, annots in annotations.items():
        task = Task(config, annots, cuid)
        device = torch.device("cuda")
        task.get_query(config, device)


if __name__ == "__main__":

    debug = True
    mode = "val"
    annotation_path = os.path.join('./DLCV_vq2d_data', 'vq_{}.json'.format(mode))
    with open(annotation_path) as fp:
        annotations = json.load(fp)
    clipwise_annotations_list = eval_utils.custom_convert_annotations_to_clipwise_list(annotations)

    if debug:
        config.debug = True
        clips_list = list(clipwise_annotations_list.keys())
        clips_list = sorted([c for c in clips_list if c is not None])
        clips_list = clips_list[0: 1]
        clipwise_annotations_list = {
            k: clipwise_annotations_list[k] for k in clips_list
        }
    perform_query(clipwise_annotations_list, config)