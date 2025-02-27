import os
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
from queue import Empty as QueueEmpty

import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from torch import multiprocessing as mp

from config.config import config, update_config
from utils import exp_utils
from eval_utils import eval_utils
from eval_utils.task_inference_clip import Task
from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher
import clip


class WorkerWithDevice(mp.Process):
    def __init__(self, config, task_queue, results_queue, worker_id, device_id, model):
        self.config = config
        self.device_id = device_id
        self.worker_id = worker_id
        super().__init__(target=self.work, args=(task_queue, results_queue))
        self.model = model

    def work(self, task_queue, results_queue):

        device = torch.device(f"cuda:{self.device_id}")

        while True:
            try:
                task = task_queue.get(timeout=1.0)
            except QueueEmpty:
                break
            key_name = task.run(self.config, device,self.model)
            results_queue.put(key_name)
            del task


def get_results(annotations, config, model):
    num_gpus = torch.cuda.device_count()
    mp.set_start_method("forkserver")

    task_queue = mp.Queue()
    for cuid, annots in annotations.items():
        task = Task(config, annots, cuid)
        task_queue.put(task)
    # Results will be stored in this queue
    results_queue = mp.Queue()

    num_processes = 1 #num_gpus

    pbar = tqdm.tqdm(
        desc=f"Get RT results",
        position=0,
        total=len(annotations),
    )

    workers = [
        WorkerWithDevice(config, task_queue, results_queue, i, i % num_gpus, model)
        for i in range(num_processes)
    ]
    # Start workers
    for worker in workers:
        worker.start()
    # Update progress bar
    predicted_rts = {}
    n_completed = 0
    while n_completed < len(annotations):
        pred = results_queue.get()
        predicted_rts.update(pred)
        n_completed += 1
        pbar.update()
    # Wait for workers to finish
    for worker in workers:
        worker.join()
    pbar.close()
    return predicted_rts

# TODO: 12/18 to our format
def format_predictions(annotations, predicted_rts):
    # Format predictions
    # predictions = {
    #     "version": annotations["version"],
    #     "challenge": "ego4d_vq2d_challenge",
    #     "results": {"videos": []},
    # }
    predictions = {}
    
    for cuid, c in annotations.items():
        predictions[cuid] = {"predictions":[]}
        for idx, a in enumerate(c["annotations"]):
            auid = idx
            apred = {
                "query_sets": {},
            }
            for qid in a["query_sets"].keys():
                if (auid, qid) in predicted_rts[cuid]:
                    rt_pred = predicted_rts[cuid][(auid, qid)][0].to_json()
                    apred["query_sets"][qid] = rt_pred
                else:
                    apred["query_sets"][qid] = {"bboxes": [], "score": 0.0}

            predictions[cuid]["predictions"].append(apred)
       
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Train hand reconstruction network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        "--eval", dest="eval", action="store_true",help="evaluate model")
    parser.add_argument(
        "--debug", dest="debug", action="store_true",help="evaluate model")
    parser.add_argument(
        "--gt-fg", dest="gt_fg", action="store_true",help="evaluate model")
    parser.add_argument(
        "--inference_path", type=str, default=""
    )
    parser.add_argument(
        "--output_path", type=str, default="val_results.json"
    )
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


if __name__ == '__main__':
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    mode = 'eval' if args.eval else 'val'
    config.inference_cache_path = os.path.join(output_dir, f'inference_cache_{mode}')
    
    if args.inference_path != "":
        config.inference_cache_path = args.inference_cache_path
    
    os.makedirs(config.inference_cache_path, exist_ok=True)
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    mode = 'test_unannotated' if args.eval else 'val'
    annotation_path = os.path.join('./DLCV_vq2d_data', 'vq_{}.json'.format(mode))
    with open(annotation_path) as fp:
        annotations = json.load(fp)
    clipwise_annotations_list = eval_utils.custom_convert_annotations_to_clipwise_list(annotations)


    test_id = ["88faa83e-e8dc-4df5-a7a1-a6158aefa17b"]

    if args.debug:
        if test_id == []:
            clips_list = list(clipwise_annotations_list.keys())
            clips_list = sorted([c for c in clips_list if c is not None])
            clips_list = clips_list[1: 2]
            clipwise_annotations_list = {
                k: clipwise_annotations_list[k] for k in clips_list
            }
        else:
            clips_list = test_id
            clipwise_annotations_list = {
                k: clipwise_annotations_list[k] for k in clips_list
            }
    
    model, preprocess = clip.load("ViT-B/32", device="cuda")
    
    predictions_rt = get_results(clipwise_annotations_list, config, model)
    # print(predictions_rt)
    # print(len(predictions_rt))
    if args.debug:
        annotations = {cuid:annotations[cuid] for cuid in clips_list}
        predictions = format_predictions(annotations, predictions_rt)
        print(predictions)
    else:
        predictions = format_predictions(annotations, predictions_rt)
    if not args.debug:
        with open(args.output_path, 'w') as fp:
            json.dump(predictions, fp)